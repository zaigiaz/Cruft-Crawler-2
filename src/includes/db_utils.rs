use crate::includes::file_utils::File_Meta;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use sled::{Batch, open};
use std::io::SeekFrom;

/// State of our Sled Database containing the current open Database, and a backup batch in case of actor failure
pub struct DbState {
    backup_batch: Vec<File_Meta>,
    database: sled::Db,
}


impl DbState {

    /// initialize and open the database
    pub fn new(path: &str) -> Result<Self, Box<dyn Error>> {
      let db = sled::open(path)
            .map_err(|e| format!("Couldn't open database at {}: {}", path, e))?;

        Ok(DbState {
            backup_batch: Vec::new(),
            database: db,
        })
    }


    /// add db entry given key and value pair
    pub fn db_add(&self, key: i32, value: &File_Meta, batch: &mut Batch) -> Result<(), Box<dyn Error>> {

	// serialise struct into u8
	let value_s = value.to_bytes()?;

	// serialize i32 to bytes
	let key_s = key.to_be_bytes();

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
    pub fn db_edit(&self, key: i32, value: File_Meta, batch: &mut Batch) -> Result<(), Box<dyn Error>> {
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

}
