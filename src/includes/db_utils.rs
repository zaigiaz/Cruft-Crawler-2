#[allow(dead_code)]
use crate::includes::file_utils::File_Meta;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use sled::{Batch, open};
use std::io::SeekFrom;


struct db_state {
    db_id: i32,
    backup_batch: Vec<File_Meta>,
}


// initialize and open the database
pub fn open_db(path: &str) -> sled::Db {

  let db: sled::Db = sled::open(path).expect("couldnt open db");
 
  db
}


// add db entry given key and value pair
pub fn db_add(key: i32, value: &File_Meta, batch: &mut Batch) -> Result<(), Box<dyn Error>> {

    // serialise struct into u8
    let value_s = value.to_bytes()?;

    // serialize i32 to bytes
    let key_s = key.to_be_bytes();

    batch.insert(&key_s, value_s);
    // let _insert = db.insert(key_s, value_s)?;

Ok(())
}


// remove db entry given key
pub fn db_remove(key: i32, batch: &mut Batch) -> Result<(), Box<dyn Error>> {
    let key_s = key.to_be_bytes();   
    batch.remove(&key_s);
    Ok(())
}


// edit db entry given key
pub fn db_edit(key: i32, value: File_Meta, batch: &mut Batch) -> Result<(), Box<dyn Error>> {
    // sled has immutable db, so we need to delete old key then insert new
    let _ = db_remove(key, batch)?;
    let _ = db_add(key, &value, batch)?;
    Ok(())
}
