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

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, 'uploads/'); 
    },
    filename: (req, file, cb) => {
      const extname = path.extname(file.originalname);
      const filename = Date.now() + '.mp4'; // we need to include not just mp4s
      cb(null, filename);
    },
  });

const upload = multer({ storage: storage });

// console.log(process.env.OPENAI_API_KEY);
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,  // DM ME FOR API KEY
});

app.post('/transcribe', upload.single('file'), async (req, res) => {
  const file = req.file;

  if (!file) {
    return res.status(400).send('No file uploaded.');
  }

  if (file.mimetype !== 'video/mp4') {
    console.log("wrong")
  }

  const filePath = path.join(path.resolve(), file.path);

  try {

    const stats = fs.statSync(filePath);

    if (stats.isFile()) {
      console.log('It is a file.');
      console.log(stats)
    } else if (stats.isDirectory()) {
      console.log('It is a directory.');
    } else {
      console.log('It is something else.');
    }

    // console.log('OpenAI Client Initialized:', openai)
    // async function testOpenAI() {
    //     const completion = await openai.chat.completions.create({
    //         model: 'gpt-3.5-turbo',  // Use a valid model like 'gpt-3.5-turbo'
    //         messages: [{ role: 'user', content: 'Hello, how are you?' }],
    //       });
    //     console.log("hi")
    //     console.log(completion.choices[0].message)
    // }

    // testOpenAI()
    async function transcribe() {
        const f = (fs.createReadStream(filePath))
        const transcription = await openai.audio.transcriptions.create({
            file: f,
            model: 'whisper-1',
        });
        console.log(transcription.text)

        const completion = await openai.chat.completions.create({
            model: 'gpt-3.5-turbo',
            messages: [{
                role: 'user',
                content: `Create quiz questions based on the following audio transcription. The questions should test whether the person answering listened to the audio:\n\n${transcription.text}`,
            }],
            });
        const questions = (completion.choices[0].message.content)
        console.log(questions)

        res.json({ transcription: transcription.text, questions: questions });

        fs.unlinkSync(filePath);
    }
    transcribe()

  } catch (error) {
    console.error('Error during transcription:', error);
    res.status(500).send('Error transcribing video.');
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
