import React, { useState } from 'react';
import Legend from './components/Legend';
import LabelItem from './types/LabelItem';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [segmentedImage, setSegmentedImage] = useState<string | null>(null);
  const [cellTypeCountTable, setCellTypeCountTable] = useState<LabelItem[]>([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(e.target.files ? e.target.files[0] : null);
    setSegmentedImage(null);
    setCellTypeCountTable([]);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('/segment', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Error uploading file');
      }

      const data = await response.json();
      console.log(data);
      if (data.segmented_image) {
        setSegmentedImage(data.segmented_image);
      }
      if (data.total_cell_count) {
        setCellTypeCountTable([{ name: 'Nuclei', color: [0, 0, 0], count: data.total_cell_count }]);
      } else if (Array.isArray(data.cell_type_count_table)) {
        setCellTypeCountTable(data.cell_type_count_table);
      } else {
        setCellTypeCountTable([]);
      }
    } catch (error) {
      console.error(error);
      alert('Something went wrong, please try again!');
    }
  };

  return (
    <div style={{ margin: '40px' }}>
      <h1>Cell Nuclei Segmentation</h1>
      <input type="file" onChange={handleFileChange} accept="image/*" />
      <button onClick={handleUpload} disabled={!selectedFile}>
        Segment
      </button>

      {segmentedImage && (
        <div style={{ marginTop: '20px' }}>
          <h2>Segmented Image</h2>
          <img
            src={`data:image/png;base64,${segmentedImage}`}
            alt="Segmented result"
            style={{ maxWidth: '500px', border: '1px solid #ccc' }}
          />
        </div>
      )}

      {cellTypeCountTable.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <h2>Cell Type Counts</h2>
          <Legend items={cellTypeCountTable} />
        </div>
      )}
    </div>
  );
}

export default App;
