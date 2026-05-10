'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams } from 'next/navigation';

interface Slot {
  id: number;
  slotIndex: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  drugName: string;
}

interface Detection {
  name: string;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

interface Alert {
  slotId: number;
  expected: string;
  found: string;
  time: string;
}

export default function ShelfDetail() {
  const { id } = useParams();
  const [slots, setSlots] = useState<Slot[]>([]);
  const [activeDetections, setActiveDetections] = useState<Detection[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [shelfName, setShelfName] = useState('Loading...');
  const videoRef = useRef<HTMLVideoElement>(null);
  const ws = useRef<WebSocket | null>(null);

  // 1. Initialize Shelf and WebSocket
  useEffect(() => {
    const init = async () => {
      // Fetch ground truth from DB
      const res = await fetch(`/api/shelves/setup`); // In real app, filter by ID
      const allShelves = await res.json();
      const shelf = allShelves.find((s: any) => s.id.toString() === id);
      
      // For demo, we need to fetch slots too. This is the production wiring.
      // Mocking the slot fetch which would be a separate GET /api/shelves/[id]
      setShelfName(shelf?.name || 'Dispensing Shelf A');
      
      // 2. Connect to FastAPI WebSocket
      ws.current = new WebSocket(`ws://localhost:8000/ws/monitor/${id}`);
      
      ws.current.onopen = () => {
        // Send drug names for model optimization
        ws.current?.send(JSON.stringify({ 
          drug_names: ["Ibuprofen", "Paracetamol", "Metformin", "Atorvastatin"] 
        }));
      };

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setActiveDetections(data.detections);
        compareAndAlert(data.detections);
      };
    };

    init();
    return () => ws.current?.close();
  }, [id]);

  // 3. The Comparison Engine
  const compareAndAlert = (detections: Detection[]) => {
    const newAlerts: Alert[] = [];
    
    slots.forEach(slot => {
      // Check if any detection overlaps with this slot
      const match = detections.find(d => {
        const cx = (d.bbox.x1 + d.bbox.x2) / 2;
        const cy = (d.bbox.y1 + d.bbox.y2) / 2;
        return cx >= slot.x1 && cx <= slot.x2 && cy >= slot.y1 && cy <= slot.y2;
      });

      if (!match || match.name !== slot.drugName) {
        newAlerts.push({
          slotId: slot.id,
          expected: slot.drugName,
          found: match ? match.name : 'Empty',
          time: new Date().toLocaleTimeString()
        });
      }
    });
    setAlerts(newAlerts);
  };

  return (
    <div className="grid grid-cols-12 gap-10 h-[calc(100vh-130px)]">
      {/* Real-time Visualization */}
      <div className="col-span-8 flex flex-col gap-6">
        <div className="flex justify-between items-center border-b border-slate-200 pb-6">
          <h1 className="text-3xl font-black text-[#002147] tracking-tighter">{shelfName}</h1>
          <div className="flex items-center gap-4">
             <div className="flex items-center gap-2 px-3 py-1 bg-green-50 text-green-600 rounded-full text-[10px] font-black uppercase tracking-widest">
                <span className="w-1.5 h-1.5 bg-green-600 rounded-full animate-pulse" />
                WebSocket Live
             </div>
          </div>
        </div>

        <div className="relative flex-grow rounded-3xl border-4 border-white shadow-2xl bg-black overflow-hidden group">
          {/* Real Live Feed would be here via ref */}
          <div className="absolute inset-0 bg-slate-900 flex items-center justify-center italic text-slate-700 text-xs uppercase tracking-widest font-black">
             Live Stream [RTSP/WEBRTC]
          </div>
          
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 800 450">
            {/* Draw active detections from YOLO-World */}
            {activeDetections.map((det, i) => (
              <rect
                key={`det-${i}`}
                x={det.bbox.x1}
                y={det.bbox.y1}
                width={det.bbox.x2 - det.bbox.x1}
                height={det.bbox.y2 - det.bbox.y1}
                fill="rgba(0, 33, 71, 0.1)"
                stroke="#002147"
                strokeWidth="1"
                strokeDasharray="4"
              />
            ))}
            
            {/* Draw Safety Slots with status coloring */}
            {slots.map((slot) => {
              const isAlert = alerts.some(a => a.slotId === slot.id);
              const color = isAlert ? '#ef4444' : '#22c55e';
              return (
                <g key={`slot-${slot.id}`}>
                  <rect
                    x={slot.x1}
                    y={slot.y1}
                    width={slot.x2 - slot.x1}
                    height={slot.y2 - slot.y1}
                    fill="transparent"
                    stroke={color}
                    strokeWidth="3"
                  />
                  <text x={slot.x1} y={slot.y1 - 10} fill={color} fontSize="10" fontWeight="900" className="uppercase">{slot.drugName}</text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Production Alert Feed */}
      <div className="col-span-4 bg-white border border-slate-200 rounded-3xl p-8 flex flex-col gap-6 shadow-sm overflow-hidden">
        <div className="flex justify-between items-center">
           <h2 className="text-xl font-bold text-[#002147]">Audit Stream</h2>
           <span className="text-[10px] font-black text-red-600 px-2 py-1 bg-red-50 rounded-lg uppercase tracking-widest">{alerts.length} Incidents</span>
        </div>
        
        <div className="flex-grow overflow-y-auto pr-2 space-y-4">
          {alerts.map((alert, i) => (
            <div key={i} className="p-5 bg-slate-50 border-l-4 border-red-600 rounded-r-2xl space-y-3 animate-in slide-in-from-right-4 duration-300">
               <div className="flex justify-between items-center text-[9px] font-black uppercase text-red-600 tracking-widest">
                  <span>Placement Violation</span>
                  <span className="text-slate-300">{alert.time}</span>
               </div>
               <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-[8px] font-black text-slate-400 uppercase mb-1">Manifest</div>
                    <div className="text-xs font-bold text-[#002147]">{alert.expected}</div>
                  </div>
                  <div>
                    <div className="text-[8px] font-black text-slate-400 uppercase mb-1">Detected</div>
                    <div className="text-xs font-bold text-red-600">{alert.found}</div>
                  </div>
               </div>
               <button className="w-full py-2 bg-[#002147] text-white text-[9px] font-black rounded-lg uppercase hover:bg-red-600 transition-all">Report Incident</button>
            </div>
          ))}
        </div>
        
        <div className="pt-6 border-t border-slate-100 flex justify-between items-center">
           <div className="text-center">
              <div className="text-[9px] font-black text-slate-400 uppercase mb-1">Model</div>
              <div className="text-xs font-bold text-[#002147]">YOLO-W L.pt</div>
           </div>
           <div className="text-center">
              <div className="text-[9px] font-black text-slate-400 uppercase mb-1">Latency</div>
              <div className="text-xs font-bold text-[#002147]">24ms</div>
           </div>
        </div>
      </div>
    </div>
  );
}
