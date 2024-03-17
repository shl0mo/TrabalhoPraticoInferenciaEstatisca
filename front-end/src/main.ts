import './style.css'

async function downloadDataset(datasetUrl: string): Promise<void> {
  const parts = datasetUrl.split('/');
  const datasateName = parts.pop() || 'dataset'; // Ajuste para evitar undefined
  const datasetId = parts[parts.length - 1];

  const downloadUrl = `https://archive.ics.uci.edu/ml/machine-learning-databases/${datasetId}/${datasateName}.zip`;
  console.log('Downloading dataset:', downloadUrl);
  try {
    const response = await fetch(downloadUrl);
    const blob = await response.blob();

    const handle = await window.showSaveFilePicker();
    const writable = await handle.createWritable();
    await writable.write(blob);
    await writable.close();

    console.log('Download complete!');
  } catch (err) {
    console.error('Download failed:', err);
  }
}

document.getElementById('download-form')?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const datasetUrl = (document.getElementById('dataset-url') as HTMLInputElement).value;
  await downloadDataset(datasetUrl);
});
