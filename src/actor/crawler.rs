#![allow(unused)]

use steady_state::*;

use std::path::{Path, PathBuf};
use std::io::prelude::*;
use std::error::Error;

use crate::includes::file_utils::*;
use crate::includes::write_ahead_log::*;

// TODO: fallback logic if entire program crashes (or if files already in DB)
// TODO: cleanup crate names and prune redundancies

// TODO: think about how this should work: fields, etc.
pub(crate) struct CrawlerState {
    // pub(crate) crawler_iter: Path,
    pub(crate) abs_path:  PathBuf,    
}

// run function 
pub async fn run(actor: SteadyActorShadow, crawler_tx: SteadyTx<File_Meta>, 
                 state: SteadyState<CrawlerState>) -> Result<(),Box<dyn Error>> {

    let actor = actor.into_spotlight([], [&crawler_tx]);

	if actor.use_internal_behavior {
	    internal_behavior(actor, crawler_tx, state).await
	} else {
	    actor.simulated_behavior(vec!(&crawler_tx)).await
	}
}


// Internal behaviour for the actor
async fn internal_behavior<A: SteadyActor>(mut actor: A, crawler_tx: SteadyTx<File_Meta>,
                                           state: SteadyState<CrawlerState>) -> Result<(),Box<dyn Error>> {

    // lock state and tx channel
    let mut state = state.lock(|| CrawlerState{abs_path: PathBuf::new()}).await;
    let mut crawler_tx = crawler_tx.lock().await;

    // TODO: replace this with config file or setup at command line
    // let search_path = read_config();
    let crawl_path = Path::new("/home/zaigiaz/Programming/Cruft-Crawler-2/src/test_directory/another/");

    let vec_metadata: Vec<File_Meta> = visit_dir(crawl_path, &mut state)?;
    
    // ai model code was sending here

    while actor.is_running(|| crawler_tx.mark_closed()) {

	actor.wait_vacant(&mut crawler_tx, 1).await;

	for m in &vec_metadata {
	    let message = m.clone();	  
	    actor.try_send(&mut crawler_tx, message).expect("couldn't send to DB");
	}

	// TODO: implement voting or consensus logic	
	actor.request_shutdown().await;
    }

	return Ok(());
}


// TODO: finish unit testing for crawler
#[cfg(test)]
pub(crate) mod crawler_tests {

    use steady_state::*;
    use super::*;

#[test]
fn test_crawler() -> Result<(), Box<dyn Error>> {

    let mut graph = GraphBuilder::for_testing().build(());
    let (crawler_tx, crawler_rx)                   = channel_builder.build();
    let state = new_state();


    graph.actor_builder().with_name("UnitTest")
        .build(move |context| internal_behavior(actor, crawler_tx, state), SoloAct);



    // TODO: test sending on crawler_tx
    // assert_steady_tx_eq_send!(&crawler_tx, File_Meta struct)

    // TODO: test file_hash function
    let test_str: PathBuf = "./src/test_directory/second.txt";
    let test_output: String = String::from("example");
    let hashed_output: String = visit_dir(test_str);
    assert_eq!(hashed_output, test_output);


    // TODO: test visit_dir function
    // test_crawl = visit_dir("test directory path")

    
    graph.start();
    // because clean shutdown waits for closed and empty
    // , it does not happen until our test data is digested. 
    graph.request_shutdown(); // critical before block_until_stopped
    Ok(())
	}
}

