const fs = require('fs');
const https = require('https');

function downloadDataset(datasetUrl) {
  // Extract the dataset ID from the URL
  const parts = datasetUrl.split('/');
  
  const datasateName = datasetUrl.split('/').pop();
  const datasetId = parts[parts.length - 2];


  // Construct the download URL
  const downloadUrl = `https://archive.ics.uci.edu/static/public/${datasetId}/${datasateName}.zip`;

  // print url
  console.log(downloadUrl);


  // Download the file
  https.get(downloadUrl, (response) => {
    const file = fs.createWriteStream(`${datasateName}.zip`);
    response.pipe(file);

    file.on('finish', () => {
      file.close();
      console.log('Download complete!');
    });
  }).on('error', (err) => {
    console.error('Download failed:', err);
  });
}

// Example usage
const datasetUrl = 'https://archive.ics.uci.edu/dataset/53/iris';
downloadDataset(datasetUrl);