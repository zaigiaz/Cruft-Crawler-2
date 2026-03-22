#![allow(unused)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use steady_state::*;
use crate::includes::file_utils::File_Meta;
use crate::includes::db_utils::*;
use crate::includes::config::*;

use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use sled::Batch;
// NOTE: add check to make sure counter is always asc order for unit testing

// TODO: use .back() to get iter for last element, then compare with write-ahead log to ensure we are at correct position
// NOTE: addtionally I can think of this like a play-cursor or iterator, can be used in addition with inotify actor later

// last id we had in our state

// size of batch we want (# of File_Meta Structs before writing to DB)
const BATCH_SIZE: usize = 10;

pub async fn run(actor: SteadyActorShadow, 
                 crawler_rx: SteadyRx<File_Meta> ) -> Result<(),Box<dyn Error>> {

    internal_behavior(actor.into_spotlight([&crawler_rx], []), crawler_rx).await
}


async fn internal_behavior<A: SteadyActor>(mut actor: A,
                                           crawler_rx: SteadyRx<File_Meta>) -> Result<(),Box<dyn Error>> {

    let mut crawler_rx = crawler_rx.lock().await;

    // TODO: add surefire pathway to database
    // TODO: add way to get last key from db | can use .back()
    let db = DbState::new("/home/zaigiaz/Programming/Cruft-Crawler-2/data/db")?;

    // TODO: scan db and get last key for this    
    let iter: sled::Iter;

    let mut loop_ctr: i32 = 0;
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
	let msg = recieved.expect("expected File_Meta Struct (crawler -> db_actor)");
	msg.meta_print();


	// NOTE: db_loop counter could also be used to check how many prompts we have given to the LLM so far,
	// to reduce context window by reprompting and wiping past history in LLM
	loop_ctr += 1;
	db_id    += 1;

	write_log("/home/zaigiaz/Programming/Cruft-Crawler-2/data/write_ahead_log.txt", db_id, msg.clone());

	// I want the db to have: db_id (counter), db_hash: key is hash value of file, prompt addition message as content;
	let _add = DbState::db_add(&db, db_id, &msg, &mut batch);
	}

	// apply batch to db (this is atomic and prevents failure in case actor failure during operation)
	db.apply_batch(batch)?;
	loop_ctr = 0;	
    }

  Ok(())
}
