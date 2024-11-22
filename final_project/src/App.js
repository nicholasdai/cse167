import React from 'react';
import "@cloudscape-design/global-styles/index.css"
import Table from '@cloudscape-design/components/table'; // Correct import
import { SpaceBetween } from '@cloudscape-design/components'; // Import SpaceBetween component for layout

const columns = [
  {
    id: 'name',
    header: 'Name',
    cell: item => item.name, // Use `cell` function to render each cell in this column
  },
  {
    id: 'age',
    header: 'Age',
    cell: item => item.age, // `cell` function to render each cell in this column
  },
  {
    id: 'country',
    header: 'Country',
    cell: item => item.country, // `cell` function to render each cell in this column
  },
];

const items = [
  { name: 'John Doe', age: 28, country: 'USA' },
  { name: 'Jane Smith', age: 34, country: 'Canada' },
  { name: 'Bob Johnson', age: 45, country: 'UK' },
];

function App() {
  return (
    <SpaceBetween direction="vertical" size="l">
      <Table
        columnDefinitions={columns}
        items={items}
        trackBy="name"
        loading={false}
      />
    </SpaceBetween>
  );
}

export default App;
