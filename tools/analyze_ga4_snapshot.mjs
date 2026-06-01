#!/usr/bin/env node

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

function parseArgs(argv) {
  const args = {
    input: null,
    output: "site/.seo/ga4-snapshot-analysis.md",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--output" || arg === "-o") {
      args.output = argv[++i];
    } else if (arg === "--help" || arg === "-h") {
      args.help = true;
    } else if (!args.input) {
      args.input = arg;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  return args;
}

function parseCsvLine(line) {
  const cells = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cell += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      cells.push(cell);
      cell = "";
    } else {
      cell += ch;
    }
  }

  cells.push(cell);
  return cells;
}

function parseDate(raw) {
  if (!/^\d{8}$/.test(raw)) return raw;
  return `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`;
}

function parseSnapshot(text) {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/);
  const sections = [];
  let startDate = null;
  let endDate = null;
  let i = 0;

  while (i < lines.length) {
    const trimmed = lines[i].trim();

    if (trimmed.startsWith("# Start date:")) {
      startDate = parseDate(trimmed.replace("# Start date:", "").trim());
      i += 1;
      continue;
    }

    if (trimmed.startsWith("# End date:")) {
      endDate = parseDate(trimmed.replace("# End date:", "").trim());
      i += 1;
      continue;
    }

    if (!trimmed || trimmed.startsWith("#")) {
      i += 1;
      continue;
    }

    const header = parseCsvLine(lines[i]);
    const rows = [];
    i += 1;

    while (i < lines.length) {
      const rowLine = lines[i];
      const rowTrimmed = rowLine.trim();
      if (!rowTrimmed || rowTrimmed.startsWith("#")) break;
      rows.push(parseCsvLine(rowLine));
      i += 1;
    }

    sections.push({ header, rows });
  }

  return { startDate, endDate, sections };
}

function sectionByHeader(snapshot, expected) {
  return snapshot.sections.find((section) => (
    expected.every((field, index) => section.header[index] === field)
  ));
}

function numberCell(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function integer(value) {
  return Math.round(value).toLocaleString("en-US");
}

function tableRows(section) {
  if (!section) return [];
  return section.rows.map((row) => Object.fromEntries(
    section.header.map((field, index) => [field, row[index] ?? ""]),
  ));
}

function topRows(rows, key, limit) {
  return [...rows]
    .sort((a, b) => numberCell(b[key]) - numberCell(a[key]))
    .slice(0, limit);
}

function markdownTable(headers, rows) {
  if (rows.length === 0) return "_No rows._";
  const escape = (value) => String(value).replace(/\|/g, "\\|");
  return [
    `| ${headers.map(escape).join(" | ")} |`,
    `| ${headers.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${headers.map((header) => escape(row[header] ?? "")).join(" | ")} |`),
  ].join("\n");
}

function pageUrlHint(title) {
  const normalized = title.toLowerCase();
  if (normalized.startsWith("stepan zolotukhin")) return "/";
  if (normalized.startsWith("blog ")) return "/blog/";
  if (normalized.startsWith("about ")) return "/about/";
  if (normalized.startsWith("zinc documentation")) return "/zinc/docs/";
  if (normalized.startsWith("zinc benchmarks")) return "/zinc/benchmarks";
  if (normalized.startsWith("zinc ") && normalized.includes("llm inference")) return "/zinc";
  if (normalized.includes("not found")) return "/404";
  return "";
}

function buildReport(snapshot, inputPath) {
  const totals = tableRows(sectionByHeader(snapshot, [
    "Active users",
    "New users",
    "Average engagement time per active user",
    "Event count",
  ]))[0] ?? {};

  const pages = tableRows(sectionByHeader(snapshot, [
    "Page title and screen class",
    "Views",
    "Active users",
    "Event count",
    "Bounce rate",
  ])).map((row) => ({
    title: row["Page title and screen class"],
    url: pageUrlHint(row["Page title and screen class"]),
    views: numberCell(row.Views),
    users: numberCell(row["Active users"]),
    events: numberCell(row["Event count"]),
    bounce: numberCell(row["Bounce rate"]),
    viewsPerUser: numberCell(row.Views) / Math.max(1, numberCell(row["Active users"])),
  }));

  const firstUserSources = tableRows(sectionByHeader(snapshot, [
    "First user source / medium",
    "Active users",
  ]));

  const sessionSources = tableRows(sectionByHeader(snapshot, [
    "Session source / medium",
    "Sessions",
  ]));

  const cities = tableRows(sectionByHeader(snapshot, [
    "City",
    "Active users",
  ]));

  const highTraffic = topRows(pages, "views", 12).map((page) => ({
    Page: page.title,
    URL: page.url,
    Views: integer(page.views),
    Users: integer(page.users),
    "Bounce rate": percent(page.bounce),
  }));

  const highBounce = pages
    .filter((page) => page.views >= 25 && page.bounce >= 0.45)
    .sort((a, b) => (b.bounce - a.bounce) || (b.views - a.views))
    .slice(0, 12)
    .map((page) => ({
      Page: page.title,
      URL: page.url,
      Views: integer(page.views),
      Users: integer(page.users),
      "Bounce rate": percent(page.bounce),
    }));

  const stickyPages = pages
    .filter((page) => page.views >= 30 && page.bounce <= 0.2)
    .sort((a, b) => (a.bounce - b.bounce) || (b.views - a.views))
    .slice(0, 10)
    .map((page) => ({
      Page: page.title,
      URL: page.url,
      Views: integer(page.views),
      "Bounce rate": percent(page.bounce),
      "Views/user": page.viewsPerUser.toFixed(1),
    }));

  const sourceTotal = firstUserSources.reduce(
    (sum, row) => sum + numberCell(row["Active users"]),
    0,
  );
  const firstUserSourceRows = topRows(firstUserSources, "Active users", 10).map((row) => ({
    Source: row["First user source / medium"],
    Users: integer(numberCell(row["Active users"])),
    Share: percent(numberCell(row["Active users"]) / Math.max(1, sourceTotal)),
  }));

  const sessionTotal = sessionSources.reduce(
    (sum, row) => sum + numberCell(row.Sessions),
    0,
  );
  const sessionSourceRows = topRows(sessionSources, "Sessions", 10).map((row) => ({
    Source: row["Session source / medium"],
    Sessions: integer(numberCell(row.Sessions)),
    Share: percent(numberCell(row.Sessions) / Math.max(1, sessionTotal)),
  }));

  const cityRows = topRows(cities, "Active users", 10).map((row) => ({
    City: row.City,
    Users: integer(numberCell(row["Active users"])),
  }));

  const organicUsers = firstUserSources
    .filter((row) => row["First user source / medium"].includes("google / organic"))
    .reduce((sum, row) => sum + numberCell(row["Active users"]), 0);

  const directUsers = firstUserSources
    .filter((row) => row["First user source / medium"] === "(direct) / (none)")
    .reduce((sum, row) => sum + numberCell(row["Active users"]), 0);

  const notFound = pages.find((page) => page.title.toLowerCase().includes("not found"));
  const totalUsers = numberCell(totals["Active users"]);

  const actionItems = [
    `Export Google Search Console next. This GA4 file has traffic and engagement, but not search queries, impressions, CTR, or ranking position.`,
    `Prioritize organic acquisition: Google organic is ${integer(organicUsers)} first-user visits (${percent(organicUsers / Math.max(1, totalUsers))}) while direct is ${integer(directUsers)} (${percent(directUsers / Math.max(1, totalUsers))}).`,
  ];

  if (notFound) {
    actionItems.push(`Fix 404 discovery. The Not Found page has ${integer(notFound.views)} views from ${integer(notFound.users)} users with ${percent(notFound.bounce)} bounce; inspect Cloudflare/hosting logs or GA4 page paths to identify broken URLs.`);
  }

  if (highBounce.length > 0) {
    actionItems.push(`Rewrite intros and internal-link blocks on the high-bounce pages below. Keep the query intent above the fold and add next-step links to /zinc, docs, benchmarks, and relevant posts.`);
  }

  actionItems.push(`Keep expanding the pages with low bounce. They are proven entry points; add clearer title tags, richer snippets, and internal links from related posts.`);

  return `# GA4 SEO Snapshot Analysis

Source: \`${inputPath}\`

Period: ${snapshot.startDate ?? "unknown"} to ${snapshot.endDate ?? "unknown"}

Generated: ${new Date().toISOString()}

## Summary

- Active users: ${integer(totalUsers)}
- New users: ${integer(numberCell(totals["New users"]))}
- Average engagement time per active user: ${numberCell(totals["Average engagement time per active user"]).toFixed(1)}s
- Event count: ${integer(numberCell(totals["Event count"]))}
- Google organic first-user traffic: ${integer(organicUsers)} users (${percent(organicUsers / Math.max(1, totalUsers))})

## What To Do Next

${actionItems.map((item, index) => `${index + 1}. ${item}`).join("\n")}

## Top Pages

${markdownTable(["Page", "URL", "Views", "Users", "Bounce rate"], highTraffic)}

## High-Bounce Opportunities

${markdownTable(["Page", "URL", "Views", "Users", "Bounce rate"], highBounce)}

## Sticky Pages To Amplify

${markdownTable(["Page", "URL", "Views", "Bounce rate", "Views/user"], stickyPages)}

## First-User Sources

${markdownTable(["Source", "Users", "Share"], firstUserSourceRows)}

## Session Sources

${markdownTable(["Source", "Sessions", "Share"], sessionSourceRows)}

## Top Cities

${markdownTable(["City", "Users"], cityRows)}

## Missing Data

This export does not contain page paths, search queries, impressions, CTR, or average Google position. Pull Search Console Performance data grouped by \`page + query + date + device\` to decide which title tags, meta descriptions, and content clusters to change first.
`;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.input) {
    console.log("Usage: node tools/analyze_ga4_snapshot.mjs <ga4-export.csv> [--output site/.seo/ga4-snapshot-analysis.md]");
    process.exit(args.help ? 0 : 1);
  }

  const inputPath = resolve(args.input);
  const outputPath = resolve(args.output);
  const snapshot = parseSnapshot(readFileSync(inputPath, "utf8"));
  const report = buildReport(snapshot, inputPath);

  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, report);
  console.log(`Wrote ${outputPath}`);
}

main();
