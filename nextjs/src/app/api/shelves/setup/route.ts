import { NextResponse } from 'next/server';
import { db } from '@/db';
import { shelves, slots, drugs } from '@/db/schema';

export async function POST(req: Request) {
  try {
    const { shelfName, cameraId, drugNames, imageBase64 } = await req.json();

    // 1. Call FastAPI for initial detection
    const mlResponse = await fetch('http://localhost:8000/detect', {
      method: 'POST',
      body: JSON.stringify({ drug_names: drugNames, image: imageBase64 }),
    }).then(r => r.json());

    // 2. Sort detections left-to-right, top-to-bottom
    const sorted = mlResponse.detections.sort((a: any, b: any) =>
      a.bbox.y1 - b.bbox.y1 || a.bbox.x1 - b.bbox.x1
    );

    // 3. Create shelf in DB
    const [shelf] = await db.insert(shelves)
      .values({ name: shelfName, cameraId })
      .returning();

    // 4. Write slots and drugs
    for (let i = 0; i < sorted.length; i++) {
      const det = sorted[i];
      const [slot] = await db.insert(slots)
        .values({ 
            shelfId: shelf.id, 
            slotIndex: i + 1, 
            x1: det.bbox.x1, 
            y1: det.bbox.y1, 
            x2: det.bbox.x2, 
            y2: det.bbox.y2 
        })
        .returning();
        
      await db.insert(drugs)
        .values({ 
            slotId: slot.id, 
            shelfId: shelf.id, 
            drugName: det.name 
        });
    }

    return NextResponse.json({ success: true, shelfId: shelf.id, slotsCreated: sorted.length });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
