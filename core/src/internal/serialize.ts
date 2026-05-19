// JSON (de)serialization with a schema-version stamp. All public kinocat data
// structures serialize to plain JSON (numbers/arrays/strings) — no class
// instances, no Map on the wire (spec §15.8).

export const SCHEMA_VERSION = 1;

export interface Versioned<T> {
  schemaVersion: number;
  data: T;
}

export function toJSON<T>(data: T): string {
  const wrapped: Versioned<T> = { schemaVersion: SCHEMA_VERSION, data };
  return JSON.stringify(wrapped);
}

export function fromJSON<T>(json: string): T {
  const parsed = JSON.parse(json) as Versioned<T>;
  if (parsed.schemaVersion !== SCHEMA_VERSION) {
    throw new Error(
      `kinocat: schema version mismatch (got ${parsed.schemaVersion}, expected ${SCHEMA_VERSION})`,
    );
  }
  return parsed.data;
}
