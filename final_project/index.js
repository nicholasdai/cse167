// Import the built-in http module
const http = require('http');

// Create a server that responds with an HTML page displaying "Hello World!" in large text
const server = http.createServer((req, res) => {
  // Set the response content type to HTML
  res.setHeader('Content-Type', 'text/html');

  // Respond with HTML that displays "Hello World!" in large font
  res.end(`
    <html>
      <head>
        <style>
          body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin-top: 20%;
            font-size: 100px;
          }
        </style>
      </head>
      <body>
        <h1>Hello World!</h1>
      </body>
    </html>
  `);
});

// Define the port number
const PORT = 3000;

// Start the server and log the URL to the console
server.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
