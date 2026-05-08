#!/usr/bin/env node
// Generate .webp next to every .png under docs/**/images/.
// Skip files whose .webp counterpart is already up-to-date.

import { readdir } from 'node:fs/promises'
import { existsSync, statSync } from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import sharp from 'sharp'

const QUALITY = 82
const ROOT = path.resolve('docs')

async function* walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  for (const entry of entries) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      yield* walk(full)
    } else if (/\.png$/i.test(entry.name)) {
      yield full
    }
  }
}

async function main() {
  let converted = 0
  let skipped = 0
  for await (const png of walk(ROOT)) {
    if (!/[\\/]images[\\/]/.test(png)) continue
    const webp = png.replace(/\.png$/i, '.webp')
    if (existsSync(webp) && statSync(webp).mtimeMs >= statSync(png).mtimeMs) {
      skipped++
      continue
    }
    await sharp(png).webp({ quality: QUALITY }).toFile(webp)
    converted++
    console.log(`webp: ${path.relative(process.cwd(), webp)}`)
  }
  console.log(`png-to-webp: ${converted} converted, ${skipped} up-to-date`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
