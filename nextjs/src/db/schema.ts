import {
  pgTable,
  serial,
  text,
  timestamp,
  integer,
  real,
  boolean,
} from 'drizzle-orm/pg-core';

export const shelves = pgTable('shelves', {
  id: serial('id').primaryKey(),
  name: text('name').notNull(),
  cameraId: text('camera_id').notNull(),
  locationName: text('location_name'),
  createdAt: timestamp('created_at').defaultNow(),
});

export const slots = pgTable('slots', {
  id: serial('id').primaryKey(),
  shelfId: integer('shelf_id').references(() => shelves.id),
  slotIndex: integer('slot_index').notNull(), // 1,2,3... left-to-right top-to-bottom
  x1: real('x1').notNull(),
  y1: real('y1').notNull(),
  x2: real('x2').notNull(),
  y2: real('y2').notNull(),
  rowNumber: integer('row_number'),
  colNumber: integer('col_number'),
});

export const drugs = pgTable('drugs', {
  id: serial('id').primaryKey(),
  slotId: integer('slot_id').references(() => slots.id),
  shelfId: integer('shelf_id').references(() => shelves.id),
  drugName: text('drug_name').notNull(),
  isEmpty: boolean('is_empty').default(false),
  lastUpdated: timestamp('last_updated').defaultNow(),
});

export const alerts = pgTable('alerts', {
  id: serial('id').primaryKey(),
  shelfId: integer('shelf_id').references(() => shelves.id),
  slotId: integer('slot_id').references(() => slots.id),
  expectedDrug: text('expected_drug').notNull(),
  foundDrug: text('found_drug').notNull(),
  triggeredAt: timestamp('triggered_at').defaultNow(),
  resolved: boolean('resolved').default(false),
  resolvedAt: timestamp('resolved_at'),
});
