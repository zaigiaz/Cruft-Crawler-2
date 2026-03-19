#![allow(unused)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use steady_state::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::Write;
use crate::actor::crawler::FileMeta;
use std::path::{Path, PathBuf};
use sled::{Batch, open};
use std::io::SeekFrom;

// #[macro_use]
// use crate::utils;

// TODO: add check to make sure counter is always asc order
// NOTE: for unit testing

// TODO: use .back() to get iter for last element, then compare with write-ahead log to ensure we are at correct position
// NOTE: addtionally I can think of this like a play-cursor or iterator, can be used in addition with inotify actor later

// last id we had in our state
struct db_state {
    db_id: i32,
    backup_batch: Vec<FileMeta>,
}

// size of batch we want (# of FileMeta Structs before writing to DB)
const BATCH_SIZE: usize = 10;

pub async fn run(actor: SteadyActorShadow, 
                 crawler_rx: SteadyRx<FileMeta> ) -> Result<(),Box<dyn Error>> {

    internal_behavior(actor.into_spotlight([&crawler_rx], []), crawler_rx).await
}


async fn internal_behavior<A: SteadyActor>(mut actor: A,
                                           crawler_rx: SteadyRx<FileMeta>) -> Result<(),Box<dyn Error>> {

    let mut crawler_rx = crawler_rx.lock().await;

    // TODO: add surefire pathway to database
    // TODO: add way to get last key from db | can use .back()
    let mut db: sled::Db = sled::open("./data/db").expect("couldnt open db");

    let iter: sled::Iter;
    let mut loop_ctr: i32 = 0;

    // TODO: scan db and get last key for this    
    let mut db_id: i32 = 0;


    while actor.is_running(|| crawler_rx.is_closed_and_empty()) {

	let mut batch = Batch::default();
	let mut batch_size = 0;
	let unit_cnt  = actor.avail_units(&mut crawler_rx);

	await_for_any!(actor.wait_avail(&mut crawler_rx, BATCH_SIZE),
		       actor.wait_timeout(Duration::from_secs(5)));
	

	if crawler_rx.avail_units() >= BATCH_SIZE {
	    batch_size = BATCH_SIZE;
	} else {
	    batch_size = crawler_rx.avail_units();
	}
	
	while loop_ctr < batch_size as i32 { 

	let recieved = actor.try_take(&mut crawler_rx);
	let msg = recieved.expect("expected FileMeta Struct (crawler -> db_actor)");
	msg.meta_print();


	// NOTE: db_loop counter could also be used to check how many prompts we have given to the LLM so far,
	// to reduce context window by reprompting and wiping past history in LLM
	loop_ctr += 1;
	db_id    += 1;

	write_log("./data/write_ahead_log.txt", db_id, msg.clone());

	// I want the db to have: db_id (counter), db_hash: key is hash value of file, prompt addition message as content;
	let _add = db_add(db_id, &msg, &mut batch);
	}

	// apply batch to db (this is atomic and prevents failure in case actor failure during operation)
	db.apply_batch(batch)?;
	loop_ctr = 0;	
    }

  Ok(())
}


// add db entry given key and value pair
fn db_add(key: i32, value: &FileMeta, batch: &mut Batch) -> Result<(), Box<dyn Error>> {

    // serialise struct into u8
    let value_s = value.to_bytes()?;

    // serialize i32 to bytes
    let key_s = key.to_be_bytes();

    batch.insert(&key_s, value_s);
    // let _insert = db.insert(key_s, value_s)?;

Ok(())
}


// edit db entry given key
fn db_edit(key: i32, value: FileMeta, batch: &mut Batch) -> Result<(), Box<dyn Error>> {
    // sled has immutable db, so we need to delete old key then insert new
    let _ = db_remove(key, batch)?;
    let _ = db_add(key, &value, batch)?;
    Ok(())
}


// remove db entry given key
fn db_remove(key: i32, batch: &mut Batch) -> Result<(), Box<dyn Error>> {
    let key_s = key.to_be_bytes();   
    batch.remove(&key_s);
    Ok(())
}


// TODO: Better write ahead logic and log cleanup (rotate log?)
fn write_log(WriteFile: &str, key: i32, value: FileMeta) -> Result<(), std::io::Error> {


    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(WriteFile)?;

    let value = value.abs_path.to_string_lossy();

    writeln!(file, "{} {}", key, value).map_err(|e| {
        eprintln!("Couldn't write to file: {}", e);
        std::io::Error::new(std::io::ErrorKind::Other, e)
    })?;

    Ok(())
}


// check our WAL and then see if current batch is different from the last that was written to file
// if so then do all operations that arent in DB and update until we reach back to current data
fn check_log(Readfile: &str) -> Result<(), std::io::Error> {

    let mut file = OpenOptions::new()
	                       .read(true)
	                       .open(Readfile)?;


    // start at beginning of file.
    // TODO: check entirety of WAL and DB for inconsistency
    // TODO: think about structure of the Crawler iterator and how interact with WAL and DB
    file.seek(SeekFrom::Start(0))?;
    



	Ok(())
}

