import React, { useState } from 'react';
import axios from 'axios';
import "@cloudscape-design/global-styles/index.css";
import { Container, SpaceBetween, Button, Flashbar } from '@cloudscape-design/components';

function App() {
  const [videoUrl, setVideoUrl] = useState(null);
  const [error, setError] = useState(null);
  const [transcription, setTranscription] = useState('No transcription');
  const [questions, setQuestions] = useState('No questions');

  // Handle file selection and preview
  const handleFileChange = async (event) => {
    const file = event.target.files[0];

    if (file && file.type === 'video/mp4') {
      const videoUrl = URL.createObjectURL(file);
      setVideoUrl(videoUrl);
      setError(null); // Clear any previous errors

      // Send the file to the server for transcription
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await axios.post('http://localhost:3001/transcribe', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        console.log(response)
        setTranscription(response.data.transcription); // Display transcription
        setQuestions(response.data.questions); // Display transcription

      } catch (err) {
        setError('Error transcribing video.');
      }
    } else {
      setError('Please upload a valid MP4 file.');
    }
  };

  return (
    <Container>
      <SpaceBetween direction="vertical" size="l">
        <h1>Sample Question Generator</h1>

        {/* File Input */}
        <div>
          <input
            type="file"
            accept="video/mp4"
            onChange={handleFileChange}
          />
        </div>

        {/* Error Message */}
        {error && (
          <Flashbar items={[{ header: error, type: 'error' }]} />
        )}

        {/* Video Player */}
        {videoUrl && (
          <div>
            <video controls width="600">
              <source src={videoUrl} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        )}

        {/* Transcription Text */}
        {transcription && (
          <div>
            <p>{transcription}</p>
          </div>
        )}

        {/* Sample Questions */}
        {questions && (
          <div>
            <p>{questions}</p>
          </div>
        )}
      </SpaceBetween>
    </Container>
  );
}

export default App;
