use crate::includes::file_utils::FileMetadata;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use sled::{Batch, open};
use std::io::SeekFrom;

/// backup batch in case of actor failure
/// note that sled already has WAL support so crash redundancy is already handled for us
/// NOTE :: do we need trees in the struct?
pub struct DbState {
    db_path: PathBuf, 
    database: sled::Db,
    backup_batch: Vec<FileMetadata>,
}


/// ------------------------------
/// TODO :: Implement methods for the two trees now
/// NOTE :: file_tree, hash_tree
/// ------------------------------
impl DbState {

    /// initialize and open the database
    pub fn open(path: PathBuf) -> Result<Self, Box<dyn Error>> {
      let db = sled::open(&path)
            .map_err(|e| format!("Couldn't open database at {:?}: {}", path, e))?;


	let file_tree = db.open_tree("file_tree").expect("open file_path tree");
	let hash_tree = db.open_tree("hash_tree").expect("open hash_values tree");

        Ok(DbState {
	    db_path: path,
	    database: db,
            backup_batch: Vec::new(),	   
        })
    }



    /// add db entry given key and value pair
    pub fn db_add(&self, key: i32, value: &FileMetadata, batch: &mut Batch) -> Result<(), Box<dyn Error>> {

	// serialise struct into u8
	let key_s = key.to_be_bytes();
	let value_s = value.to_bytes()?;


	// serialize i32 to bytes
	batch.insert(&key_s, value_s);
	// let _insert = db.insert(key_s, value_s)?;

	Ok(())
    }


    /// remove db entry given key
    pub fn db_remove(&self, key: i32, batch: &mut Batch) -> Result<(), Box<dyn Error>> {
	let key_s = key.to_be_bytes();   
	batch.remove(&key_s);
	Ok(())
    }


    /// edit db entry given key
    pub fn db_edit(&self, key: i32, value: FileMetadata, batch: &mut Batch) -> Result<(), Box<dyn Error>> {
	// sled has immutable db, so we need to delete old key then insert new
	self.db_remove(key, batch)?;
	self.db_add(key, &value, batch)?;
	Ok(())
    }

    /// Flush the batch to the database
    pub fn apply_batch(&self, batch: Batch) -> Result<(), Box<dyn Error>> {
        self.database.apply_batch(batch)?;
        Ok(())
    }


   /// re-opens database after crash
   pub fn recover(&mut self) ->  Result<sled::Db, Box<dyn std::error::Error>> {
        
        let new_DB_state =  DbState::open(self.db_path.clone())?;
        self.database = new_DB_state.database;
        
        // backup_batch is restored from checkpoint
        // file_tree and hash_tree are recovered by sled
              
        Ok(self.database.clone())
   }

}

