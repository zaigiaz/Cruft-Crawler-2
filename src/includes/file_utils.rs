use steady_state::StateGuard;
use std::io::prelude::*;
use walkdir::{WalkDir,DirEntry, IntoIter};
use std::ffi::OsStr;
use filetime::FileTime;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use std::error::Error;
use sha2::{Sha256, Digest};
use hex;

use crate::actor::crawler::CrawlerState;


/// metadata struct for all the variables to make decisions with
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct FileMetadata {
    pub abs_path:  String,
    pub file_name: String,
    pub hash:      String,
    pub is_file:   bool,
    pub size:      u64,
    pub modified:  i64,
    pub created:   i64,
    pub readonly:  bool,
} 

impl FileMetadata {
    /// for easy debugging of struct if needed
   pub fn meta_print(&self) {
        println!("\n--------------------");
	println!("Absolute_Path: {:?}", self.abs_path);
	println!("File_Name: {}",       self.file_name);
	println!("hash: {}",            self.hash);
	println!("is_file: {}",         self.is_file);
	println!("size: {}",            self.size);
	println!("modified: {}",        self.modified);
	println!("created: {}",         self.created);
	println!("read-only: {}",       self.readonly);
        println!("--------------------");
    }

    /// serialize into bytes using bincode
    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn Error>> {
	Ok(serde_cbor::to_vec(self)?)
    }

    /// deserialize from bytes using bincode
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
	Ok(serde_cbor::from_slice(bytes)?)
    }
}



/// takes in a Walkdir Iterator and returns the filtered metadata struct for it
pub fn get_file_metadata(entry: DirEntry) -> Result<FileMetadata, Box<dyn Error>> {

    let abs_path: String = entry.path()
	.to_path_buf().to_string_lossy().to_string();

    // state.abs_path = abs_path.clone();
    
    let name_os: &OsStr = entry.file_name();
    let file_name: String = match name_os.to_str() {
	Some(s) => s.to_owned(),
	None => name_os.to_string_lossy().into_owned(),
    };
    
    let meta = entry.metadata()?;

    let is_file:  bool   = meta.is_file();
    let size:     u64    = meta.len();
    let modified: i64    = FileTime::from_last_modification_time(&meta).seconds() / 60;
    let created:  i64    = FileTime::from_creation_time(&meta).expect("created file time").seconds() / 60;
    let readonly: bool   = meta.permissions().readonly();
    let mut hash: String = String::new();

    if meta.is_file() {
	hash = get_file_hash(abs_path.clone()).expect("didn't get hash value");
    } else { hash = String::from(""); }
    
    let new_meta = FileMetadata {
	abs_path,
        file_name,
	hash, 
        is_file,
        size,
        modified, 
        created,
        readonly,
    };

    Ok(new_meta)
}


/// filter for the walkdir iterator that can skip directories and other specified files
pub fn our_filter(entry: &DirEntry) -> bool {

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
pub fn get_file_hash(file_name: String) -> Result<String, Box<dyn Error>> {

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
