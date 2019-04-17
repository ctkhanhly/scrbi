//const request = require('request');
const {PythonShell} = require('python-shell');
const {Storage} = require('@google-cloud/storage');
//import {PythonShell} from 'python-shell';

PythonShell.run('handTrack.py', null, (err, results)=>{
    if(err) throw err;
    //array of print statements
    console.log(results);
});

//trackingImage.png

//https://cloud.google.com/nodejs/docs/reference/storage/2.3.x/

//https://github.com/googleapis/nodejs-storage/blob/master/samples/files.js

async function uploadFile(bucketName, filename) {
// [START storage_upload_file]
// Imports the Google Cloud client library

//const {Storage} = require('@google-cloud/storage');

// Creates a client
const storage = new Storage();

/**
 * TODO(developer): Uncomment the following lines before running the sample.
 */
// const bucketName = 'Name of a bucket, e.g. my-bucket';
// const filename = 'Local file to upload, e.g. ./local/path/to/file.txt';

// Uploads a local file to the bucket
await storage.bucket(bucketName).upload(filename, {
    // Support for HTTP requests made with `Accept-Encoding: gzip`
    gzip: true,
    // By setting the option `destination`, you can change the name of the
    // object you are uploading to a bucket.
    metadata: {
    // Enable long-lived HTTP caching headers
    // Use only if the contents of the file will never change
    // (If the contents will change, use cacheControl: 'no-cache')
    cacheControl: 'public, max-age=31536000',
    },
});

console.log(`${filename} uploaded to ${bucketName}.`);
// [END storage_upload_file]
}


async function deleteFile(bucketName, filename) {
    // [START storage_delete_file]
    // Imports the Google Cloud client library
    
    //const {Storage} = require('@google-cloud/storage');
  
    // Creates a client
    const storage = new Storage();
  
    /**
     * TODO(developer): Uncomment the following lines before running the sample.
     */
    // const bucketName = 'Name of a bucket, e.g. my-bucket';
    // const filename = 'File to delete, e.g. file.txt';
  
    // Deletes the file from the bucket
    await storage
      .bucket(bucketName)
      .file(filename)
      .delete();
  
    console.log(`gs://${bucketName}/${filename} deleted.`);
    // [END storage_delete_file]
  }



//EXECUTION
uploadFile("tracking-images", "trackingImage.png");
//do gg api functions here
//now delete trackingImage.png to run this code next time

deleteFile("tracking-images", "trackingImage.png")