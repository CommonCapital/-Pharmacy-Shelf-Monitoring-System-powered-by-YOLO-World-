'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';

interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  name: string;
}

export default function SetupPage() {
  const [image, setImage] = useState<string | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [currentBox, setCurrentBox] = useState<Partial<Box> | null>(null);
  const [drugName, setDrugName] = useState('');
  const [shelfName, setShelfName] = useState('');
  const [cameraId, setCameraId] = useState('');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const router = useRouter();

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  const startDrawing = (e: React.MouseEvent) => {
    if (!image) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    setCurrentBox({ x1: e.clientX - rect.left, y1: e.clientY - rect.top });
  };

  const endDrawing = (e: React.MouseEvent) => {
    if (!currentBox || !image) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x2 = e.clientX - rect.left;
    const y2 = e.clientY - rect.top;
    
    const name = prompt("Enter Drug Name for this slot:");
    if (name) {
      setBoxes([...boxes, { ...currentBox as any, x2, y2, name }]);
    }
    setCurrentBox(null);
  };

  useEffect(() => {
    if (image && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const img = new Image();
      img.src = image;
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx?.drawImage(img, 0, 0);
        
        // Draw existing boxes
        boxes.forEach(box => {
          if (ctx) {
            ctx.strokeStyle = '#002147';
            ctx.lineWidth = 4;
            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            ctx.fillStyle = '#002147';
            ctx.fillRect(box.x1, box.y1 - 25, 120, 25);
            ctx.fillStyle = 'white';
            ctx.fillText(box.name, box.x1 + 5, box.y1 - 8);
          }
        });
      };
    }
  }, [image, boxes]);

  const saveShelf = async () => {
    // Implementation to call /api/shelves/setup with manual boxes
    const res = await fetch('/api/shelves/setup', {
      method: 'POST',
      body: JSON.stringify({ shelfName, cameraId, boxes, image }),
    });
    if (res.ok) router.push('/');
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-center border-b border-slate-200 pb-6">
        <div>
          <h1 className="text-3xl font-black text-[#002147]">Shelf Calibration</h1>
          <p className="text-slate-400 text-sm">Upload photo → Draw boxes → Name drugs</p>
        </div>
        <button 
          onClick={saveShelf}
          disabled={boxes.length === 0}
          className="btn-navy disabled:opacity-50"
        >
          Finalize Setup ({boxes.length} Slots)
        </button>
      </div>

      <div className="grid grid-cols-12 gap-8">
        {/* Left: Interactive Canvas */}
        <div className="col-span-9 space-y-4">
          {!image ? (
            <label className="flex flex-col items-center justify-center w-full h-[500px] border-4 border-dashed border-slate-200 rounded-3xl cursor-pointer hover:bg-slate-50 transition-all group">
              <div className="text-slate-300 group-hover:text-[#002147] transition-colors">
                <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>
                <p className="font-bold uppercase tracking-widest text-xs">Upload Shelf Reference Image</p>
              </div>
              <input type="file" className="hidden" onChange={handleImageUpload} accept="image/*" />
            </label>
          ) : (
            <div className="relative rounded-2xl border-8 border-white shadow-2xl overflow-hidden bg-slate-200 cursor-crosshair">
              <canvas 
                ref={canvasRef}
                onMouseDown={startDrawing}
                onMouseUp={endDrawing}
                className="max-w-full h-auto"
              />
            </div>
          )}
        </div>

        {/* Right: Configuration */}
        <div className="col-span-3 space-y-6">
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
            <h3 className="font-bold text-[#002147] text-sm uppercase tracking-wider">Identity</h3>
            <input 
              placeholder="Shelf Name (e.g. Dispensing A)"
              className="w-full p-3 bg-slate-50 border border-slate-100 rounded-xl text-sm outline-none focus:border-[#002147]"
              value={shelfName}
              onChange={(e) => setShelfName(e.target.value)}
            />
            <input 
              placeholder="Camera ID"
              className="w-full p-3 bg-slate-50 border border-slate-100 rounded-xl text-sm outline-none focus:border-[#002147]"
              value={cameraId}
              onChange={(e) => setCameraId(e.target.value)}
            />
          </div>

          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="font-bold text-[#002147] text-sm uppercase tracking-wider mb-4">Detected Slots</h3>
            <div className="space-y-2 max-h-[300px] overflow-y-auto">
              {boxes.length === 0 ? (
                <p className="text-slate-300 text-xs italic">No slots defined yet...</p>
              ) : (
                boxes.map((box, i) => (
                  <div key={i} className="flex justify-between items-center p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="text-xs font-bold text-[#002147] truncate max-w-[120px]">{box.name}</span>
                    <button 
                      onClick={() => setBoxes(boxes.filter((_, idx) => idx !== i))}
                      className="text-red-500 text-[10px] font-bold"
                    >
                      Delete
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
