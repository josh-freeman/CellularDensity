import React from 'react';
import LabelItem  from '../types/LabelItem';


interface LegendProps {
  items: LabelItem[];
}

const Legend: React.FC<LegendProps> = ({ items }) => {
  return (
    <div className="border rounded-lg p-4 shadow-sm inline-block">
      {items.map((item, index) => (
        <div key={index} className="flex items-center gap-2 py-1">
          <span
            className="inline-block h-3 w-3 rounded-full"
            style={{ backgroundColor: `rgb(${item.color.join(',')})` }}
          ></span>
          <span className="font-medium">
            {item.name}: {item.count}
          </span>
        </div>
      ))}
    </div>
  );
};

export default Legend;
