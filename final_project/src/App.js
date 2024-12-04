import React, { useState, useEffect } from 'react';
import axios from 'axios';
import "@cloudscape-design/global-styles/index.css";
import { Container, SpaceBetween, Button, Flashbar, Cards, FormField } from '@cloudscape-design/components';

// subject to be changed once this is scaleable
const PORT = '3001'
const apiLink = 'http://localhost:' + PORT + '/transcribe'

function App() {

  const [videoUrl, setVideoUrl] = useState(null);
  const [error, setError] = useState(null);
  const [transcription, setTranscription] = useState('No transcription');
  const [questions, setQuestions] = useState('No questions');
  const [answers, setAnswers] = useState('No answers');

  const handleFileChange = async (file) => {

    if (file && (file.type === 'audio/mp3' || file.type === 'audio/mpeg' || file.type === 'video/mp4')) {
      setVideoUrl(URL.createObjectURL(file));

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await axios.post(apiLink, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        setTranscription(response.data.transcription);
        setQuestions(response.data.questions);
        setAnswers(response.data.answers);

      } catch (err) {
        setError('Error transcribing video: ' + err);
      }
    } else {
      setError('Please upload a valid MP3 or MP4 file.');
    }
  };

  return (
    <Container>
      <SpaceBetween direction="vertical" size="l">
        <h1>Sample Question Generator</h1>

        <FormField label="Input a File">
          <div>
            <input
              type="file"
              accept="audio/mp3,video/mp4"
              onChange={(e) => handleFileChange(e.target.files[0])}
            />
            <button onClick={() => window.location.reload()}>
              Start Over
            </button>
          </div>
        </FormField>

        {error && (
          <Flashbar items={[{ header: error, type: 'error' }]} />
        )}

        {videoUrl && (
          <div>
            <video controls width="400">
              <source src={videoUrl} type={videoUrl.endsWith('.mp4') ? 'video/mp4' : 'audio/mp3'} />
            </video>
          </div>
        )}

        {transcription && (
          <div>
            <h3>Transcription</h3>
            <p style={{ whiteSpace: 'pre-line' }}>{transcription}</p>
          </div>
        )}

        {questions && (
          <div>
            <h3>Questions</h3>
            <p style={{ whiteSpace: 'pre-line' }}>{questions}</p>
          </div>
        )}

        {answers && (
          <div>
            <h3>Answers</h3>
            <p style={{ whiteSpace: 'pre-line' }}>{answers}</p>
          </div>
        )}
      </SpaceBetween>
    </Container>
  );
}

export default App;
