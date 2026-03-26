use crate::includes::file_utils::FileMetadata;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use sled::{Batch, open, Ivec};
use std::io::SeekFrom;

/// ------------------------------
/// NOTE :: do we need trees in the struct?
/// TODO :: Implement Trait for all of this?
/// ------------------------------
pub struct DbState {
    db_path: PathBuf, 
    database: sled::Db,
    file_tree: sled::Tree,
    hash_tree: sled::Tree,
    deletion_tree: sled::Tree,
}


/// ------------------------------
/// TODO :: Implement methods for the two trees now
/// TODO :: Remove all these helper functions and just create things to serialize keys and values?
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
        })
    }



    /// add db entry given key and value pair
    pub fn db_add(&self, value: &mut FileMetadata, batch: &mut Batch) -> Result<(), Box<dyn Error>> {

	// serialise struct into u8
	let file_tree_key: &[u8] = value.abs_path.as_bytes();
	let file_tree_value: &[u8] = &value.to_bytes()?;

	// key for our hash tree
	let hash_tree_key: &[u8] = value.hash.as_bytes();

	let file_tree = self.database.open_tree("file_tree").expect("open file_path tree");
	let hash_tree = self.database.open_tree("hash_tree").expect("open hash_path tree");

	file_tree.insert(file_tree_key, file_tree_value);
	hash_tree.insert(hash_tree_key, file_tree_key);

	Ok(())
    }


    /// remove db entry given a key value
    pub fn db_remove(&self, key: &[u8]) -> Result<(), Box<dyn Error>> {

	self.database.compare_and_swap(key, )

    }


    /// edit db entry given key
    pub fn db_edit(&self, file_path: String) -> Result<(), Box<dyn Error>> {
	
	

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

