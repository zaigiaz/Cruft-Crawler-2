use steady_state::*;
use crate::includes::file_utils::FileMetadata;
use crate::includes::db_utils::*;

use std::error::Error;
use std::fs::OpenOptions;
use std::io::{prelude::*, Write};
use std::path::{Path, PathBuf};

// size of batch we want (# of File_Meta Structs before writing to DB)
const BATCH_SIZE: usize = 10;

/// run function for the database actor
pub async fn run(actor: SteadyActorShadow, ai_tx: SteadyTx<FileMetadata>,
                 crawler_rx: SteadyRx<FileMetadata> ) -> Result<(),Box<dyn Error>> {

    internal_behavior(actor.into_spotlight([&crawler_rx], [&ai_tx]), ai_tx, crawler_rx).await
}


/// internal behaviour for the database actor
async fn internal_behavior<A: SteadyActor>(mut actor: A, ai_tx: SteadyTx<FileMetadata>,
                                           crawler_rx: SteadyRx<FileMetadata>) -> Result<(),Box<dyn Error>> {

    let mut crawler_rx = crawler_rx.lock().await;
    let mut ai_tx = ai_tx.lock().await;
    
    let db_file_name: PathBuf = "/home/zaigiaz/Programming/Cruft-Crawler-2/data/db".into();

    // takes path and returns the DbState struct, for all operations
    let database = DbState::open(db_file_name)?;

    // TODO :: scan db and get last key for this, using iterator over the file_tree
    // let iter: sled::Iter;

    while actor.is_running(|| crawler_rx.is_closed_and_empty()) {

	let mut batch_size = 0;

	await_for_any!(actor.wait_avail(&mut crawler_rx, BATCH_SIZE),
		       actor.wait_timeout(Duration::from_secs(5)));
	
	if crawler_rx.avail_units() >= BATCH_SIZE {
	    batch_size = BATCH_SIZE;
	} else {
	    batch_size = crawler_rx.avail_units();
	}
	
	for _ in 0..batch_size {
	    let recieved = actor.try_take(&mut crawler_rx);
	    let mut msg = recieved.expect("expected File_Meta Struct (crawler -> db_actor)");

	    database.insert(&msg)?;

	    /// TODO :: Data is clean until sending from here
	    // send to ai_model, await for room because inference takes 2-3 minutes
	    // actor.send_async(&mut ai_tx, msg, SendSaturation::AwaitForRoom);
	    actor.try_send(&mut ai_tx, msg);
	}
    }
  Ok(())
}
