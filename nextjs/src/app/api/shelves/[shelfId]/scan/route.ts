import { NextResponse } from 'next/server';
import { db } from '@/db';
import { slots, drugs, alerts } from '@/db/schema';
import { eq } from 'drizzle-orm';

export async function GET(req: Request, { params }: { params: { shelfId: string } }) {
  try {
    const shelfId = parseInt(params.shelfId);
    const groundTruth = await db
      .select()
      .from(slots)
      .innerJoin(drugs, eq(drugs.slotId, slots.id))
      .where(eq(slots.shelfId, shelfId));
      
    return NextResponse.json({ slots: groundTruth });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function POST(req: Request, { params }: { params: { shelfId: string } }) {
  try {
    const { imageBase64 } = await req.json();
    const shelfId = parseInt(params.shelfId);

    // 1. Load ground truth templates from DB
    const groundTruth = await db
      .select()
      .from(slots)
      .innerJoin(drugs, eq(drugs.slotId, slots.id))
      .where(eq(slots.shelfId, shelfId));

    // Construct template properties for verification
    const templates = groundTruth.map(r => ({
      label: r.drugs.drugName,
      aspect_ratio: r.drugs.aspectRatio || 1.0,
      hsv_color: {
        h: r.drugs.hsvH || 0.0,
        s: r.drugs.hsvS || 0.0,
        v: r.drugs.hsvV || 0.0,
        dominant_color: r.drugs.dominantColor || 'unknown'
      }
    }));

    // 2. Call FastAPI for zero-shot object detection and color validation
    const liveDetections = await fetch('http://localhost:8000/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: imageBase64,
        templates: templates
      })
    }).then(r => r.json());

    if (liveDetections.error) {
      throw new Error(liveDetections.error);
    }

    // 3. Compare and write alerts
    const newAlerts = [];
    for (const slot of groundTruth) {
      // Find matching detection in the slot's region using coordinates
      const live = liveDetections.detections.find((d: any) => {
          const cx = (d.bbox.x1 + d.bbox.x2) / 2;
          const cy = (d.bbox.y1 + d.bbox.y2) / 2;
          return cx >= slot.slots.x1 && cx <= slot.slots.x2 && cy >= slot.slots.y1 && cy <= slot.slots.y2;
      });

      const foundName = live ? live.name : "empty";

      if (foundName !== slot.drugs.drugName) {
        const [alert] = await db.insert(alerts).values({
          shelfId,
          slotId: slot.slots.id,
          expectedDrug: slot.drugs.drugName,
          foundDrug: foundName
        }).returning();
        newAlerts.push(alert);
      }
    }

    return NextResponse.json({ alerts: newAlerts });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
