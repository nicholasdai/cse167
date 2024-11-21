import express from 'express';

const app = express();
const port = 3000;

// Serve a simple "Hello, World!" message at the root URL
app.get('/', (req, res) => {
  res.send('<h1>Hello, World!</h1>');
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});