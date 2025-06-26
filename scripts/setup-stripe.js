#!/usr/bin/env node

/**
 * Stripe Setup Script
 * 
 * This script configures Stripe integration for the LLM Finetuning Studio.
 * It uses environment variables for secure API key management.
 * 
 * Usage: node scripts/setup-stripe.js
 * 
 * Required Environment Variables:
 * - STRIPE_API_KEY: Your Stripe secret API key
 * - STRIPE_PUBLISHABLE_KEY: Your Stripe publishable key
 * - STRIPE_WEBHOOK_SECRET: Your Stripe webhook secret
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync } from 'fs';

// Load environment variables from .env file manually
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

// Simple .env parser (for demonstration - in production use dotenv package)
try {
  const envFile = readFileSync(join(projectRoot, '.env'), 'utf8');
  envFile.split('\n').forEach(line => {
    const [key, ...valueParts] = line.split('=');
    if (key && valueParts.length > 0) {
      process.env[key.trim()] = valueParts.join('=').trim();
    }
  });
} catch (error) {
  // .env file doesn't exist, will use system environment variables
}

// Validate required environment variables
const requiredEnvVars = [
  'STRIPE_API_KEY',
  'STRIPE_PUBLISHABLE_KEY'
];

const missingVars = requiredEnvVars.filter(varName => !process.env[varName]);

if (missingVars.length > 0) {
  console.error('❌ Missing required environment variables:');
  missingVars.forEach(varName => {
    console.error(`   - ${varName}`);
  });
  console.error('\nPlease copy .env.example to .env and set the required values.');
  process.exit(1);
}

// Validate API key format
const stripeApiKey = process.env.STRIPE_API_KEY;
if (!stripeApiKey.startsWith('sk_')) {
  console.error('❌ Invalid Stripe API key format. Must start with "sk_"');
  process.exit(1);
}

// Check if using test vs live keys
const isTestMode = stripeApiKey.includes('test');
const keyType = isTestMode ? 'TEST' : 'LIVE';

console.log('🔧 Stripe Configuration Setup');
console.log('===============================');
console.log(`Mode: ${keyType}`);
console.log(`API Key: ${stripeApiKey.substring(0, 7)}...${stripeApiKey.slice(-4)}`);

if (!isTestMode) {
  console.warn('⚠️  WARNING: You are using LIVE Stripe keys!');
  console.warn('   Make sure this is intentional for production environment.');
}

// Validate webhook secret if provided
if (process.env.STRIPE_WEBHOOK_SECRET) {
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!webhookSecret.startsWith('whsec_')) {
    console.warn('⚠️  Warning: Webhook secret format may be incorrect. Should start with "whsec_"');
  } else {
    console.log(`Webhook Secret: ${webhookSecret.substring(0, 10)}...${webhookSecret.slice(-4)}`);
  }
}

// Configuration object for the application
const stripeConfig = {
  apiKey: process.env.STRIPE_API_KEY,
  publishableKey: process.env.STRIPE_PUBLISHABLE_KEY,
  webhookSecret: process.env.STRIPE_WEBHOOK_SECRET,
  testMode: isTestMode
};

console.log('\n✅ Stripe configuration validated successfully!');
console.log('\nNext steps:');
console.log('1. Verify your webhook endpoints are configured in Stripe dashboard');
console.log('2. Test the integration with a small transaction');
console.log('3. Review Stripe logs for any issues');

// Export configuration for use by other modules
export default stripeConfig;