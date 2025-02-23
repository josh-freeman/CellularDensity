import React, { useState } from 'react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [segmentedImage, setSegmentedImage] = useState(null);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setSegmentedImage(null); // Reset the segmented image when a new file is chosen
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
      if (data.segmented_image) {
        // Save the base64 of the segmented image in state
        setSegmentedImage(data.segmented_image);
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
    </div>
  );
}

export default App;
