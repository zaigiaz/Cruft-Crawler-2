use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use sled::{Batch, open, IVec};

use crate::includes::file_utils::{FileMetadata, from_bytes};

/// main struct containing state for the database
/// it has the database handle and all the open namespaces for each tree
pub struct DbState {
    db_path: PathBuf, 
    database: sled::Db,
    file_tree: sled::Tree,
    hash_tree: sled::Tree,
    deletion_tree: sled::Tree,
}

/// implemented functions for DbState
impl DbState {

    /// initialize and open the database
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let db_path = path.as_ref().to_path_buf();
        let database = sled::open(&db_path)
            .map_err(|e| format!("Couldn't open database at {:?}: {}", db_path, e))?;

        let file_tree = database.open_tree("file_tree")?;
        let hash_tree = database.open_tree("hash_tree")?;
        let deletion_tree = database.open_tree("deletion_tree")?;

        Ok(Self { db_path, database, file_tree, hash_tree, deletion_tree })
    }
    
    /// given a path, return the bytes of it
    fn file_key(path: &str) -> &[u8] {
        path.as_bytes()
    }

    /// given a hash string, return it in bytes
    fn hash_key(hash: &str) -> &[u8] {
        hash.as_bytes()
    }

    pub fn insert(&self, meta: &FileMetadata) -> Result<(), Box<dyn Error>> {
        let file_key = Self::file_key(&meta.abs_path);
        let hash_key = Self::hash_key(&meta.hash);
        let value = meta.to_bytes()?;

        let mut batch = Batch::default();
        batch.insert(file_key, value);
        self.file_tree.apply_batch(batch)?; 

        self.hash_tree.insert(hash_key, file_key)?;
        Ok(())
    }


    /// Insert using a shared batch on file_tree if you want to accumulate:
    pub fn insert_with_batch(&self, meta: &FileMetadata, batch: &mut Batch) -> Result<(), Box<dyn Error>> {
        let file_key = Self::file_key(&meta.abs_path);
        let hash_key = Self::hash_key(&meta.hash);
        let value = meta.to_bytes()?;

        batch.insert(file_key, value);
        self.hash_tree.insert(hash_key, file_key)?;
        Ok(())
    }
    
    /// apply the batch on the file_tree
    pub fn apply_batch(&self, batch: Batch) -> Result<(), Box<dyn Error>> {
        self.file_tree.apply_batch(batch)?; // atomically on file_tree[web:3][web:9]
        Ok(())
    }


    /// Get the path for the file and return the metadata struct for it
    pub fn get_by_path(&self, path: &str) -> Result<Option<FileMetadata>, Box<dyn Error>> {
        let key = Self::file_key(path);
        if let Some(ivec) = self.file_tree.get(key)? {
            let meta = from_bytes(&ivec)?;
            Ok(Some(meta))
        } else {
            Ok(None)
        }
    }

    /// given the hash, return the FileMetadata struct of that hash
    pub fn get_by_hash(&self, hash: &str) -> Result<Option<FileMetadata>, Box<dyn Error>> {
        let hk = Self::hash_key(hash);
        let Some(path_bytes) = self.hash_tree.get(hk)? else {
            return Ok(None);
        };
        let path = String::from_utf8(path_bytes.to_vec())?;
        self.get_by_path(&path)
    }


    /// update the database with a new record from a metadata struct
    pub fn update(&self, new_meta: &FileMetadata) -> Result<(), Box<dyn Error>> {
        // Load existing to know old hash, if any
        if let Some(old_meta) = self.get_by_path(&new_meta.abs_path)? {
            if old_meta.hash != new_meta.hash {
                // remove old hash entry
                let old_hk = Self::hash_key(&old_meta.hash);
                self.hash_tree.remove(old_hk)?;
            }
        }

        // Upsert new record and hash mapping
        let fk = Self::file_key(&new_meta.abs_path);
        let hk = Self::hash_key(&new_meta.hash);
        let val = new_meta.to_bytes()?;

        self.file_tree.insert(fk, val)?;
        self.hash_tree.insert(hk, fk)?;
        Ok(())
    }

    
    /// Optional CAS-style edit: only update if existing value matches
    /// ------------------------------
    /// TODO :: DO I need this?
    /// ------------------------------
    pub fn compare_and_swap_path(&self, path: &str, old: Option<&FileMetadata>,
				 new: Option<&FileMetadata>) -> Result<(), Box<dyn Error>> {

        let key = Self::file_key(path);
        let old_bytes = match old {
            Some(v) => Some(v.to_bytes()?),
            None => None,
        };
        let new_bytes = match new {
            Some(v) => Some(v.to_bytes()?),
            None => None,
        };

        // Uses sled::Tree::compare_and_swap via Deref[web:1][web:9]
        let res = self.file_tree.compare_and_swap(key, old_bytes, new_bytes)?;
        match res {
            Ok(()) => Ok(()),
            Err(e) => Err(format!("Compare-and-swap failed: {:?}", e.current).into()),
        }
    }

 
    /// given a path, remove the file from both of the trees (hash and file tree)
    pub fn delete_by_path(&self, path: &str) -> Result<(), Box<dyn Error>> {
        // Need hash to clean secondary index
        if let Some(meta) = self.get_by_path(path)? {
            let fk = Self::file_key(path);
            let hk = Self::hash_key(&meta.hash);

            self.file_tree.remove(fk)?;
            self.hash_tree.remove(hk)?;
        }
        Ok(())
    }

    /// given a hash value, remove that hash from the file_tree
    pub fn delete_by_hash(&self, hash: &str) -> Result<(), Box<dyn Error>> {
        let hk = Self::hash_key(hash);
        if let Some(path_bytes) = self.hash_tree.remove(hk)? {
            let path = String::from_utf8(path_bytes.to_vec())?;
            let fk = Self::file_key(&path);
            self.file_tree.remove(fk)?;
        }
        Ok(())
    }

    /// if the database fails, recover from the failure by just re-opening the db with same handle
    pub fn recover(&mut self) -> Result<(), Box<dyn Error>> {
        let new = DbState::open(self.db_path.clone())?;
        self.database = new.database;
        self.file_tree = new.file_tree;
        self.hash_tree = new.hash_tree;
        Ok(())
    }
}

