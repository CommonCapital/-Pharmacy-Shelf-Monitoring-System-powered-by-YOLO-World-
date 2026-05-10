'use client';

import { useState, useRef, useEffect } from 'react';

interface Box {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
}

export default function PrototypePage() {
  const [image, setImage] = useState<string | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [currentBox, setCurrentBox] = useState<{x1: number, y1: number, x2: number, y2: number} | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [videoResults, setVideoResults] = useState<any[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Draw the image and boxes on the canvas
  useEffect(() => {
    if (!image || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new window.Image();
    img.onload = () => {
      // Set canvas dimensions to match image
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Draw image
      ctx.drawImage(img, 0, 0);

      const drawBox = (b: {x1: number, y1: number, x2: number, y2: number}, label: string, isPreview: boolean) => {
        const x = Math.min(b.x1, b.x2);
        const y = Math.min(b.y1, b.y2);
        const w = Math.abs(b.x2 - b.x1);
        const h = Math.abs(b.y2 - b.y1);

        ctx.strokeStyle = isPreview ? '#ff0000' : '#002147';
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, w, h);
        
        if (label) {
          ctx.fillStyle = isPreview ? '#ff0000' : '#002147';
          ctx.fillRect(x, y - 30, ctx.measureText(label).width + 20, 30);
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 16px sans-serif';
          ctx.fillText(label, x + 10, y - 10);
        }
      };

      // Draw all saved boxes
      boxes.forEach(b => drawBox(b, b.label, false));
      
      // Draw the box currently being drawn
      if (currentBox) {
        drawBox(currentBox, "Drawing...", true);
      }
    };
    img.src = image;
  }, [image, boxes, currentBox]);

  const getCanvasCoords = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    
    // Calculate the scale between the actual pixel size of the canvas and its visual size on screen
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  const onMouseDown = (e: React.MouseEvent) => {
    const coords = getCanvasCoords(e);
    if (!coords) return;
    setIsDrawing(true);
    setCurrentBox({ x1: coords.x, y1: coords.y, x2: coords.x, y2: coords.y });
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || !currentBox) return;
    const coords = getCanvasCoords(e);
    if (!coords) return;
    setCurrentBox({ ...currentBox, x2: coords.x, y2: coords.y });
  };

  const onMouseUp = (e: React.MouseEvent) => {
    if (!isDrawing || !currentBox) return;
    setIsDrawing(false);
    
    // Ignore tiny accidental clicks (less than 10x10 pixels)
    if (Math.abs(currentBox.x2 - currentBox.x1) < 10 || Math.abs(currentBox.y2 - currentBox.y1) < 10) {
      setCurrentBox(null);
      return;
    }

    const label = prompt("Label this object:");
    if (label) {
      // Normalize coordinates so x1,y1 is always top-left
      const newBox = {
        id: Math.random().toString(36).substr(2, 9),
        x1: Math.min(currentBox.x1, currentBox.x2),
        y1: Math.min(currentBox.y1, currentBox.y2),
        x2: Math.max(currentBox.x1, currentBox.x2),
        y2: Math.max(currentBox.y1, currentBox.y2),
        label
      };
      setBoxes([...boxes, newBox]);
    }
    setCurrentBox(null);
  };

  const handleDeleteBox = (id: string) => {
    setBoxes(boxes.filter(b => b.id !== id));
  };

  const handleEditBoxLabel = (id: string) => {
    const box = boxes.find(b => b.id === id);
    if (!box) return;
    const newLabel = prompt("Enter new label:", box.label);
    if (newLabel) {
      setBoxes(boxes.map(b => b.id === id ? { ...b, label: newLabel } : b));
    }
  };

  // 1. Setup Phase: Manual Labeling
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  // 2. Monitoring Phase: Video Comparison
  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || boxes.length === 0) return;
    
    setVideoUrl(URL.createObjectURL(file));
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('video', file);
    
    // Strip the 'id' field to match the backend expectations
    const backendBoxes = boxes.map(({x1, y1, x2, y2, label}) => ({x1, y1, x2, y2, label}));
    formData.append('config', JSON.stringify(backendBoxes));

    // This calls the FastAPI backend to process video against JSON config
    const res = await fetch('http://localhost:8000/process-video', {
      method: 'POST',
      body: formData
    });
    const results = await res.json();
    setVideoResults(results);
    setIsProcessing(false);
  };

  return (
    <div className="max-w-5xl mx-auto py-10 space-y-12">
      <header className="border-b border-slate-200 pb-6">
        <h1 className="text-2xl font-black text-[#002147] tracking-tight uppercase">YOLO-World Prototype</h1>
        <p className="text-slate-400 text-sm italic">Upload Image → Label → Upload Video → Compare</p>
      </header>

      {/* Step 1: Labeling */}
      <section className="space-y-4">
        <h2 className="text-sm font-bold text-[#002147] uppercase tracking-widest">Step 1: Define Ground Truth</h2>
        {!image ? (
          <input type="file" onChange={handleImageUpload} className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-[#002147] hover:file:bg-blue-100" />
        ) : (
          <div className="flex gap-6 items-start">
            {/* Left side: Canvas */}
            <div className="flex-1 relative border-4 border-white shadow-xl rounded-lg overflow-hidden cursor-crosshair">
              <canvas 
                ref={canvasRef} 
                onMouseDown={onMouseDown} 
                onMouseMove={onMouseMove}
                onMouseUp={onMouseUp}
                onMouseLeave={onMouseUp}
                className="max-w-full h-auto block"
              />
            </div>
            
            {/* Right side: Panel */}
            <div className="w-64 shrink-0 bg-white p-4 rounded-lg shadow-xl border border-slate-200 space-y-3 sticky top-4">
              <p className="text-[10px] font-bold text-slate-400 uppercase border-b pb-2">Saved Objects (JSON)</p>
              {boxes.length === 0 && <p className="text-xs text-slate-400 italic">No boxes drawn yet.</p>}
              {boxes.map((b) => (
                <div key={b.id} className="flex justify-between items-center group">
                  <span className="text-xs font-bold text-[#002147] truncate w-24" title={b.label}>{b.label}</span>
                  <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button onClick={() => handleEditBoxLabel(b.id)} className="text-[10px] text-blue-500 font-bold hover:underline">Edit</button>
                    <button onClick={() => handleDeleteBox(b.id)} className="text-[10px] text-red-500 font-bold hover:underline">Del</button>
                  </div>
                </div>
              ))}
              {boxes.length > 0 && (
                <button onClick={() => setBoxes([])} className="text-[10px] text-red-500 font-bold underline mt-2 pt-2 border-t w-full text-left">Clear All</button>
              )}
            </div>
          </div>
        )}
      </section>

      {/* Step 2: Comparison */}
      {boxes.length > 0 && (
        <section className="space-y-4 border-t border-slate-100 pt-10">
          <h2 className="text-sm font-bold text-[#002147] uppercase tracking-widest">Step 2: Upload Live Video to Compare</h2>
          <input type="file" accept="video/*" onChange={handleVideoUpload} className="block w-full text-sm text-slate-500" />
          
          {isProcessing && <div className="text-blue-600 font-bold animate-pulse text-sm mt-4">Processing Video via YOLO-World...</div>}

          {videoUrl && (
            <div className="mt-6 border-4 border-slate-100 rounded-xl overflow-hidden shadow-sm">
              <video src={videoUrl} controls className="w-full h-auto bg-black" />
            </div>
          )}
          
          {videoResults.length > 0 && (
            <div className="bg-slate-50 p-6 rounded-xl space-y-4 border border-slate-100 mt-6">
               <h3 className="text-xs font-bold text-slate-400 uppercase">Detection Diffs</h3>
               {videoResults.map((res, i) => (
                 <div key={i} className={`p-3 rounded-lg border flex justify-between items-center ${res.match ? 'bg-green-50 border-green-100' : 'bg-red-50 border-red-100'}`}>
                    <span className="text-xs font-bold">{res.label}</span>
                    <span className={`text-[10px] font-black uppercase ${res.match ? 'text-green-600' : 'text-red-600'}`}>
                       {res.match ? 'Expected match' : `Expected ${res.label} returned ${res.detected}`}
                    </span>
                 </div>
               ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
