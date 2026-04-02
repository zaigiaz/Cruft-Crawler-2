use steady_state::*;

use std::path::{Path, PathBuf};
use std::io::prelude::*;
use std::error::Error;

use crate::includes::file_utils::*;
use crate::includes::write_ahead_log::*;


/// holds last path visited of crawler actor
pub(crate) struct CrawlerState {
    pub(crate) abs_path:  String,    
}

/// run function for crawler actor
pub async fn run(actor: SteadyActorShadow, crawler_tx: SteadyTx<FileMetadata>, 
                 state: SteadyState<CrawlerState>) -> Result<(),Box<dyn Error>> {

    let actor = actor.into_spotlight([], [&crawler_tx]);

	if actor.use_internal_behavior {
	    internal_behavior(actor, crawler_tx, state).await
	} else {
	    actor.simulated_behavior(vec!(&crawler_tx)).await
	}
}


/// Internal behaviour for the crawler actor
/// ------------------------------
/// TODO :: crawler state lock update when crawling
/// TODO :: use std::env to get path of executable and get correct paths for reading and writing files
/// ------------------------------
async fn internal_behavior<A: SteadyActor>(mut actor: A, crawler_tx: SteadyTx<FileMetadata>,
                                           state: SteadyState<CrawlerState>) -> Result<(),Box<dyn Error>> {

    // lock state and tx channel
    let mut state = state.lock(|| CrawlerState{abs_path: String::new()}).await;
    let mut crawler_tx = crawler_tx.lock().await;

    // let search_path = read_config();
    // TODO :: replace this with config file or setup at command line
    let crawl_path = Path::new("/home/zaigiaz/Programming/Cruft-Crawler-2/src/");    

    // create write_ahead log for crawler, and rotate it every 5kb
    let mut write_ahead_log = RotatingLog {
	path: "/home/zaigiaz/Programming/Cruft-Crawler-2/data/write_ahead_log.txt",
	max_size: 1024 * 5,
    };
        
    // Iterator for our walkdir crate over some path
    let mut walker = walkdir::WalkDir::new(crawl_path)
	.into_iter()
	.filter_entry(|e| !our_filter(e));

    // while loop with channel mark closed detection, ends after iterator has returned last
    while actor.is_running(|| crawler_tx.mark_closed()) {

	// Read the directory (recursively, with filter)
	match walker.next() {	    
	    None => break,
	    Some(entry_res) => {

	    await_for_all!(actor.wait_vacant(&mut crawler_tx, 1));

            let entry = entry_res?;
	    
	    // skip all directories returned by iterator
	    if !entry.metadata()?.is_file() {
		continue;
	    }
	    
	    let new_metadata = get_file_metadata(entry)?;
	    
	    // debugging statement for showcase
	    // new_metadata.meta_print();
	    
	    write_ahead_log.write_log(&new_metadata.abs_path);

	    actor.try_send(&mut crawler_tx, new_metadata).expect("couldn't send to DB");
	    }
	}
    }
    
    // TODO :: restructure this loop for clean actor shutdown	
    actor.wait_shutdown().await;
    return Ok(());
}


/// Testing for the crawler actor
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



    // TODO :: test sending on crawler_tx
    // assert_steady_tx_eq_send!(&crawler_tx, File_Meta struct)

    // TODO :: test file_hash function
    let test_str: PathBuf = "./src/test_directory/second.txt";
    let test_output: String = String::from("example");
    let hashed_output: String = visit_dir(test_str);
    assert_eq!(hashed_output, test_output);


    // TODO :: test visit_dir function
    // test_crawl = visit_dir("test directory path")

    
    graph.start();
    // because clean shutdown waits for closed and empty
    // , it does not happen until our test data is digested. 
    graph.request_shutdown(); // critical before block_until_stopped
    Ok(())
	}
}

