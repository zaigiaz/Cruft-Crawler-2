use crate::includes::file_utils::File_Meta;
use std::io::{Seek, SeekFrom};
use std::fs::OpenOptions;
use std::io::Write;

// TODO: Better write ahead logic and log cleanup (rotate log?)
// TODO: make this a rotating log that keeps last batch in, then compare with iter and where it started
// TODO: then just move Iter to there from Crawler?
pub fn write_log(WriteFile: &str, key: i32, value: File_Meta) -> Result<(), std::io::Error> {


    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(WriteFile)?;

    let value = value.abs_path.to_string_lossy();

    writeln!(file, "{} {}", key, value).map_err(|e| {
        eprintln!("Couldn't write to file: {}", e);
        std::io::Error::new(std::io::ErrorKind::Other, e)
    })?;

    Ok(())
}


// check our WAL and then see if current batch is different from the last that was written to file
// if so then do all operations that arent in DB and update until we reach back to current data
pub fn check_log(Readfile: &str) -> Result<(), std::io::Error> {

    let mut file = OpenOptions::new()
	                       .read(true)
	                       .open(Readfile)?;


    // TODO: think about structure of the Crawler iterator and how interact with WAL and DB
    // TODO: check entirety of WAL and DB for inconsistency
    // start at beginning of file.
    file.seek(SeekFrom::Start(0))?;
    
	Ok(())
}
