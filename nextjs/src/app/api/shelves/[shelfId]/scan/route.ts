import { NextResponse } from 'next/server';
import { db } from '@/db';
import { slots, drugs, alerts } from '@/db/schema';
import { eq } from 'drizzle-orm';

export async function POST(req: Request, { params }: { params: { shelfId: string } }) {
  try {
    const { imageBase64 } = await req.json();
    const shelfId = parseInt(params.shelfId);

    // 1. Load ground truth from DB
    const groundTruth = await db
      .select()
      .from(slots)
      .innerJoin(drugs, eq(drugs.slotId, slots.id))
      .where(eq(slots.shelfId, shelfId));

    // 2. Call FastAPI for region-based detection
    const liveDetections = await fetch('http://localhost:8000/detect', {
      method: 'POST',
      body: JSON.stringify({
        image: imageBase64,
        drug_names: groundTruth.map(r => r.drugs.drugName)
      })
    }).then(r => r.json());

    // 3. Compare and write alerts
    const newAlerts = [];
    for (const slot of groundTruth) {
      // Find detection in the slot's region
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
