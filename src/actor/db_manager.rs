#![allow(unused)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use steady_state::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use crate::actor::crawler::FileMeta;
use std::path::{Path, PathBuf};
use sled::{Batch, open};

// TODO: Database schema
// TODO: actor shutdown?

struct db_state {
    db_id: i32,
    // more fields here
}

// size of batch we want (# of FileMeta Structs before writing to DB)
const BATCH_SIZE: usize = 2;

pub async fn run(actor: SteadyActorShadow, 
                 crawler_rx: SteadyRx<FileMeta> ) -> Result<(),Box<dyn Error>> {

    internal_behavior(actor.into_spotlight([&crawler_rx], []), crawler_rx).await
}


async fn internal_behavior<A: SteadyActor>(mut actor: A,
                                           crawler_rx: SteadyRx<FileMeta>) -> Result<(),Box<dyn Error>> {

    let mut crawler_rx = crawler_rx.lock().await;


    // TODO: example code that I need to change
    let mut db: sled::Db = sled::open("./data/db").unwrap();
    let mut loop_ctr: i32 = 0;
    let mut db_id: i32 = 0;


    // TODO: code to check db_status before doing any db operations

    while actor.is_running(|| crawler_rx.is_closed_and_empty()) {

	let mut batch = Batch::default();
	let unit_cnt  = actor.avail_units(&mut crawler_rx);
	// println!("here is {}", unit_cnt);


	// NOTE: backpressure most important part of steady state graph

	// TODO: might need to use this for timer-out operations
	// steady_await_for_all_or_proceed_upon_two()

	// TODO: nested await_for_any! macro could work to with two await_for_any!
	actor.wait_avail(&mut crawler_rx, BATCH_SIZE).await;


	// convert batch_size constant to i32 to work	
	while loop_ctr < BATCH_SIZE as i32 { 
	let recieved = actor.try_take(&mut crawler_rx);
	let msg = recieved.expect("expected FileMeta Struct (crawler -> db_actor)");

	loop_ctr += 1;
	db_id    += 1;

	write_ahead("./data/write_ahead_log.txt", db_id, msg.clone());
	let _add = db_add(db_id, &msg, &mut batch);
	// msg.meta_print();
	}

	// apply batch to db
	db.apply_batch(batch)?;

	loop_ctr = 0;

	// TODO: add check to make sure counter is always asc order
	// TODO: use .back() to get iter for last element, then compare with write-ahead log
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
    let _remove = db_remove(key, batch);
    let _add = db_add(key, &value, batch);
Ok(())
}


// remove db entry given key
fn db_remove(key: i32, batch: &mut Batch) -> Result<(), Box<dyn Error>> {
    let key_s = key.to_be_bytes();   
    batch.remove(&key_s);
    Ok(())
}


fn write_ahead(WriteFile: &str, key: i32, value: FileMeta) -> Result<(), Box<dyn Error>> {

    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(WriteFile)
        .unwrap();

    let value = value.abs_path.to_string_lossy();

    // append 
    if let Err(e) = writeln!(file, "{} {}", key, value) {
        eprintln!("Couldn't write to file: {}", e);
    }


    Ok(())
}
