#![allow(unused)]
#![allow(non_snake_case)]

use steady_state::*;
use std::error::Error;
use crate::actor::crawler::FileMeta;
use std::path::{Path, PathBuf};
use sled::{Batch, open};

// TODO: Database schema
// TODO: error handling for db functions
// TODO: actor shutdown?
// TODO: write-ahead log

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
    let mut db: sled::Db = sled::open("./src/db").unwrap();
    let mut ctr: i32 = 0;


    // TODO: code to check db_status before doing any db operations

    while actor.is_running(|| crawler_rx.is_closed_and_empty()) {

	let mut batch = Batch::default();
	actor.wait_avail(&mut crawler_rx, BATCH_SIZE).await;


	// TODO: add functionality to write to write-ahead log
	// TODO: add timer-based operations to db

	while ctr < 2 {
	let recieved = actor.try_take(&mut crawler_rx);
	let msg = recieved.expect("expected FileMeta Struct (db_actor)");

	ctr += 1;

	let _add = db_add(ctr, &msg, &mut batch);
	msg.meta_print();
	}

	db.apply_batch(batch)?;

	ctr = 0;

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


// TODO: write function
fn write_ahead(WriteFile: &Path, key: i32, value: FileMeta) -> Result<(), Box<dyn Error>> {



    Ok(())
}



