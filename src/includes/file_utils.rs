
use steady_state::StateGuard;
use std::io::prelude::*;
use walkdir::WalkDir;
use walkdir::DirEntry;
use std::ffi::OsStr;
use filetime::FileTime;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use std::error::Error;
use sha2::{Sha256, Digest};
use hex;

use crate::actor::crawler::CrawlerState;

// metadata struct
#[allow(nonstandard_style)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct File_Meta {
    pub abs_path:  PathBuf,
    pub file_name: String,
    pub hash:      String,
    pub is_file:   bool,
    pub size:      u64,
    pub modified:  i64,
    pub created:   i64,
    pub readonly:  bool,
} 

impl File_Meta {
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
    #[allow(dead_code)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
	Ok(serde_cbor::from_slice(bytes)?)
    }
}


// function to visit test directory and return metadata of each file and insert into metadata struct
// also updates state per every entry
// TODO: integration testing for visit_dir()
pub fn visit_dir(dir: &Path,
                 state: &mut StateGuard<'_, CrawlerState> ) -> Result<Vec<File_Meta>, Box<dyn Error>> {

    let mut metas: Vec<File_Meta> = Vec::new();

    let walker = WalkDir::new(dir).into_iter();

    // Read the directory (non-recursive)
    for entry_res in walker.filter_entry(|e| !our_filter(e)) {

        let entry = entry_res?;

        let abs_path: PathBuf = entry.path()
	                             .to_path_buf();

	// NOTE: update state to reflect last crawled entry
	state.abs_path = abs_path.clone();

	// new function here for metadata
	
	 
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

                metas.push(File_Meta {
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
