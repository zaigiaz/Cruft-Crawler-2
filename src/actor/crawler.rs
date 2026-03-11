#![allow(unused)]

use steady_state::*;

use std::path::{Path, PathBuf};
use sha2::{Sha256, Digest};
use std::io::prelude::*;
use walkdir::WalkDir;
use walkdir::DirEntry;
use std::ffi::OsStr;
use filetime::FileTime;
use std::error::Error;
use serde::{Serialize, Deserialize};
use hex;

// TODO: fallback logic if entire program crashes (or if files already in DB)
// TODO: cleanup crate names and prune redundancies

// TODO: think about how this should work: fields, etc.
pub(crate) struct CrawlerState {
    pub(crate) abs_path:  PathBuf,    
}

// metadata struct
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct FileMeta {
    pub abs_path:  PathBuf,
    pub file_name: String,
    pub hash:      String,
    pub is_file:   bool,
    pub size:      u64,
    pub modified:  i64,
    pub created:   i64,
    pub readonly:  bool,
} 

impl FileMeta {
// for easy debugging of struct if needed
   pub fn meta_print(&self) {
        println!("\n--------------------");
	println!("Absolute_Path: {:?}", self.abs_path);
	println!("File_Name: {}",       self.file_name);
	println!("hash: {}",            self.hash);
	println!("is_file: {}",         self.is_file);
	println!("size: {}",            self.size);
	println!("modified: {}",        self.modified / 60);
	println!("created: {}",         self.created / 60);
	println!("read-only: {}",       self.readonly);
        println!("--------------------");
    }

    // serialize into bytes using bincode
    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn Error>> {
	Ok(serde_cbor::to_vec(self)?)
    }

    // deserialize from bytes using bincode
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
	Ok(serde_cbor::from_slice(bytes)?)
    }
}


// run function 
pub async fn run(actor: SteadyActorShadow, crawler_tx: SteadyTx<FileMeta>, 
                 state: SteadyState<CrawlerState>) -> Result<(),Box<dyn Error>> {

    let actor = actor.into_spotlight([], [&crawler_tx]);

	if actor.use_internal_behavior {
	    internal_behavior(actor, crawler_tx, state).await
	} else {
	    actor.simulated_behavior(vec!(&crawler_tx)).await
	}
}


// Internal behaviour for the actor
async fn internal_behavior<A: SteadyActor>(mut actor: A, crawler_tx: SteadyTx<FileMeta>,
                                           state: SteadyState<CrawlerState>) -> Result<(),Box<dyn Error>> {

    // lock state and tx channel
    let mut state = state.lock(|| CrawlerState{abs_path: PathBuf::new()}).await;
    let mut crawler_tx = crawler_tx.lock().await;

    // TODO: replace this with config file or setup at command line
    let path1 = Path::new("/home/zaigiaz/Programming/home-lab-notes/");

    let metas: Vec<FileMeta> = visit_dir(path1, &mut state)?;
    
    // ai model code was sending here

    while actor.is_running(|| crawler_tx.mark_closed()) {

	actor.wait_vacant(&mut crawler_tx, 1).await;

	for m in &metas {
	    let message = m.clone();	  
	    actor.try_send(&mut crawler_tx, message).expect("couldn't send to DB");
	}

	// TODO: implement voting or consensus logic	
	actor.request_shutdown().await;
    }

	return Ok(());
}


// Read first 1024 bytes of file then hash, note that this hashes the bytes, not a string from the file
// TODO: double check that hashing bytes is correct (integration testing) for get_file_hash()
pub fn get_file_hash(file_name: PathBuf) -> Result<String, Box<dyn Error>> {

    let mut file = std::fs::File::open(file_name)?;

    // buffer of 1024 bytes to read file
    let mut buffer = [0u8; 1024];

    let n = file.read(&mut buffer)?;

    let mut hasher = Sha256::new();
    hasher.update(&buffer[..n]);
    let result = hasher.finalize();

    let mut out: [u8; 32] = result.into();
    out.copy_from_slice(&result);

    // encodes value as string
    let convert = hex::encode(out);

    // slice to 16 digits
    let final_value = &convert[0..16];
    
    Ok(final_value.to_string())
}


// function to visit test directory and return metadata of each file and insert into metadata struct
// also updates state per every entry
// TODO: integration testing for visit_dir()
pub fn visit_dir(dir: &Path,
                 state: &mut StateGuard<'_, CrawlerState> ) -> Result<Vec<FileMeta>, Box<dyn Error>> {

    let mut metas: Vec<FileMeta> = Vec::new();

    let walker = WalkDir::new(dir).into_iter();

    // Read the directory (non-recursive)
    for entry_res in walker.filter_entry(|e| !our_filter(e)) {

        let entry = entry_res?;

        let abs_path: PathBuf = entry.path()
	                             .to_path_buf();

	// NOTE: update state to reflect last crawled entry
	// NOTE: could use DirEntry or another concept instead
	state.abs_path = abs_path.clone();
	
 
	let name_os: &OsStr = entry.file_name();
	let file_name: String = match name_os.to_str() {
            Some(s) => s.to_owned(),
            None => name_os.to_string_lossy().into_owned(),
        };

	
        // Try to get metadata; if failing for a specific entry, skip it but continue
        match entry.metadata() {
            Ok(md) => {
                let is_file:  bool   = md.is_file();
                let size:     u64    = md.len();
                let modified: i64    = FileTime::from_last_modification_time(&md).seconds();
                let created:  i64    = FileTime::from_creation_time(&md).expect("created file time").seconds();
                let readonly: bool   = md.permissions().readonly();
		let mut hash: String = String::new();

		if is_file {
		hash = get_file_hash(abs_path.clone()).expect("didn't get hash value");
		}

                metas.push(FileMeta {
		    abs_path,
                    file_name,
		    hash, 
                    is_file,
                    size,
                    modified, 
                    created,
                    readonly,
                });
            }
            Err(e) => {
		// TODO: log errors here
                eprintln!("warning: cannot stat {}: {}", file_name, e);
            }
        }
    }
    Ok(metas)
}


// avoid hidden directories and files, other filters
// avoid linux directories that are below the home directory of the user
fn our_filter(entry: &DirEntry) -> bool {

    let filter = vec!["tmp", "var", "sys"];

    entry.file_name()
        .to_str()
          .map(|s| {
	      s.starts_with(".") || 
              filter.iter().any(|f| f.is_empty() && s.contains(f))
	  })
	  .unwrap_or(false)
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


    graph.actor_builder().with_name("UnitTest")
        .build(move |context| internal_behavior(actor, crawler_tx, state));


    graph.start();
    // because clean shutdown waits for closed and empty
    // , it does not happen until our test data is digested. 
    graph.request_shutdown(); // critical before block_until_stopped
	}
}
