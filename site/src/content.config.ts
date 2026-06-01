import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    seoTitle: z.string().optional(),
    date: z.coerce.date(),
    tags: z.array(z.string()).optional().default([]),
    keywords: z.array(z.string()).optional().default([]),
    faqs: z.array(z.object({
      question: z.string(),
      answer: z.string(),
    })).optional().default([]),
    excerpt: z.string(),
    seoDescription: z.string().optional(),
    draft: z.boolean().optional().default(false),
  }),
});

export const collections = { posts };
