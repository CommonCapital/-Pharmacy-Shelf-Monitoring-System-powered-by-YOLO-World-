import { NextResponse } from 'next/server';
import { db } from '@/db';
import { shelves, slots, drugs } from '@/db/schema';

export async function POST(req: Request) {
  try {
    const { shelfName, cameraId, boxes, image } = await req.json();

    // 1. Call FastAPI /register-templates to extract geometry & HSV color profiles
    const mlResponse = await fetch('http://localhost:8000/register-templates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: image, // base64 string
        boxes: boxes.map((b: any) => ({
          x1: b.x1,
          y1: b.y1,
          x2: b.x2,
          y2: b.y2,
          label: b.name
        }))
      })
    }).then(r => r.json());

    if (mlResponse.error) {
      throw new Error(mlResponse.error);
    }

    // 2. Create shelf in DB
    const [shelf] = await db.insert(shelves)
      .values({ name: shelfName, cameraId })
      .returning();

    // 3. Write slots and drugs with HSV templates
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      const template = mlResponse.templates ? mlResponse.templates.find((t: any) => t.label === b.name) : null;
      
      const [slot] = await db.insert(slots)
        .values({ 
            shelfId: shelf.id, 
            slotIndex: i + 1, 
            x1: b.x1, 
            y1: b.y1, 
            x2: b.x2, 
            y2: b.y2 
        })
        .returning();
        
      await db.insert(drugs)
        .values({ 
            slotId: slot.id, 
            shelfId: shelf.id, 
            drugName: b.name,
            aspectRatio: template ? template.aspect_ratio : null,
            hsvH: template ? template.hsv_color.h : null,
            hsvS: template ? template.hsv_color.s : null,
            hsvV: template ? template.hsv_color.v : null,
            dominantColor: template ? template.hsv_color.dominant_color : null
        });
    }

    return NextResponse.json({ success: true, shelfId: shelf.id, slotsCreated: boxes.length });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
