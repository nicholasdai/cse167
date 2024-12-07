import React, { useState, useEffect } from 'react';
import axios from 'axios';
import "@cloudscape-design/global-styles/index.css";
import { Container, SpaceBetween, Button, Flashbar, FormField } from '@cloudscape-design/components';

// subject to be changed once this is scaleable
const PORT = '3001'
const apiLink =  `http://localhost:${PORT}/transcribe`

function App() {
  const [videoUrl, setVideoUrl] = useState(null);
  const [error, setError] = useState(null);
  const [warning, setWarning] = useState(null);
  const [transcription, setTranscription] = useState('No transcription');
  const [questions, setQuestions] = useState('No questions');
  const [answers, setAnswers] = useState('No answers');
  const [fileName, setFileName] = useState('');
  const [visibleAnswers, setVisibleAnswers] = useState({});  // Track visibility of answers

  const handleFileChange = async (file) => {
    setError(null)

    if (file && (file.type === 'audio/mp3' || file.type === 'audio/mpeg' || file.type === 'video/mp4')) {
      setVideoUrl(URL.createObjectURL(file));
      setFileName(file.name)

      if (file.size > 5000000) {
        setWarning("Warning: This is a large file. This may take a while.")
      }

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

  const handleAnswerClick = (index) => {
    setVisibleAnswers(prevState => ({
      ...prevState,
      [index]: !prevState[index],
    }));
  };

  const downloadFile = (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
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

        {warning && (
          <Flashbar items={[{ header: warning, type: 'warning' }]} />
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
            <div>
              {questions.split('\n').map((question, index) => (
                <div key={index}>
                  <Button variant="link" onClick={() => handleAnswerClick(index)}>
                    {question}
                  </Button>
                  {visibleAnswers[index] && answers.split('\n')[index] && (
                    <p style={{ whiteSpace: 'pre-line', marginTop: '10px' }}>
                      {answers.split('\n')[index]}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        <div>
          <button onClick={() => downloadFile(`Transcription:\n${transcription}`, `${fileName}_transcription.txt`)}>
              Download Transcription
          </button>
          <button onClick={() => downloadFile(`Questions:\n${questions}\n\nAnswers:\n${answers}`, `${fileName}_questions_and_answers.txt`)}>
              Download Questions and Answers
          </button>
        </div>
      </SpaceBetween>
    </Container>
  );
}

export default App;
