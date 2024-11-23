import { OpenAI } from 'openai';
import express from 'express';
import multer from 'multer';
import path from 'path';
import { dirname } from 'path';
import fs from 'fs';
import dotenv from 'dotenv'
import cors from 'cors'

// Load environment variables from .env file
dotenv.config();

const app = express();
app.use(cors());
const port = 3001;

// Set up multer to handle file uploads
const upload = multer({ dest: 'uploads/' });

// Set up OpenAI API client
// console.log(process.env.OPENAI_API_KEY);
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,  // Get the API key from .env file
});

// API route for transcribing uploaded videos
app.post('/transcribe', upload.single('file'), async (req, res) => {
  const file = req.file;

  if (!file) {
    return res.status(400).send('No file uploaded.');
  }

  if (file.mimetype !== 'video/mp4') {
    console.log("wrong")
  }

  // Read the file
  const filePath = path.join(path.resolve(), file.path);

  try {
    // OpenAI Whisper API expects a file buffer, so we'll read it
    // console.log(filePath)

    const stats = fs.statSync(filePath);

    if (stats.isFile()) {
      console.log('It is a file.');
      console.log(stats)
    } else if (stats.isDirectory()) {
      console.log('It is a directory.');
    } else {
      console.log('It is something else.');
    }

    const f = (fs.createReadStream(filePath))

    // console.log('OpenAI Client Initialized:', openai)
    async function testOpenAI() {
        const completion = await openai.chat.completions.create({
            model: 'gpt-3.5-turbo',  // Use a valid model like 'gpt-3.5-turbo'
            messages: [{ role: 'user', content: 'Hello, how are you?' }],
          });
        console.log(completion)
    }

    testOpenAI()


    // const transcription = await openai.audio.transcriptions.create({
    //   file: f,
    //   model: 'whisper-1',
    //   response_format: "text",
    // });

    // // Clean up the uploaded file after processing
    // fs.unlinkSync(filePath);

    // res.json({ transcription: transcription.text });
  } catch (error) {
    console.error('Error during transcription:', error);
    res.status(500).send('Error transcribing video.');
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
