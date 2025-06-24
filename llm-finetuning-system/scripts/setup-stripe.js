#!/usr/bin/env node
const Stripe = require('stripe');
const { PrismaClient } = require('@prisma/client');
const stripe = require('stripe')('YOUR_STRIPE_SECRET_KEY_HERE');
const prisma = new PrismaClient();

async function setupStripeProducts() {
  console.log('🚀 Setting up Stripe products and prices...');

  const plans = [
    {
      name: 'starter',
      displayName: 'Starter',
      description: 'Ideal for individual researchers and small projects',
      price: 2900, // $29.00 in cents
      trainingCredits: 15,
      maxGpuType: 'A100',
      maxTrainingTime: 7200,
      features: [
        '15 training jobs per month',
        'A100 GPU access',
        'Models up to 13B parameters',
        'Email support',
        'Extended training time (2 hours max)',
        'Advanced monitoring',
        'Custom datasets',
        'LoRA & QLoRA support'
      ]
    },
    {
      name: 'pro',
      displayName: 'Professional',
      description: 'For teams and advanced research projects',
      price: 9900, // $99.00 in cents
      trainingCredits: 60,
      maxGpuType: 'H100',
      maxTrainingTime: 14400,
      features: [
        '60 training jobs per month',
        'H100 GPU access',
        'Models up to 70B parameters',
        'Priority support',
        'Extended training time (4 hours max)',
        'Advanced monitoring & analytics',
        'Custom datasets & models',
        'All fine-tuning techniques',
        'API access',
        'Team collaboration'
      ]
    },
    {
      name: 'enterprise',
      displayName: 'Enterprise',
      description: 'Custom solutions for large organizations',
      price: 49900, // $499.00 in cents
      trainingCredits: 300,
      maxGpuType: 'H200',
      maxTrainingTime: 86400,
      features: [
        '300 training jobs per month',
        'Latest GPU access (H200, B200)',
        'Unlimited model sizes',
        'Dedicated support team',
        'Unlimited training time',
        'Custom monitoring solutions',
        'On-premise deployment',
        'Custom integrations',
        'SLA guarantees',
        'Advanced security features'
      ]
    }
  ];

  try {
    for (const plan of plans) {
      console.log(`Creating product for ${plan.displayName}...`);

      // Create Stripe product
      const product = await stripe.products.create({
        name: plan.displayName,
        description: plan.description,
        metadata: {
          plan_name: plan.name,
          training_credits: plan.trainingCredits.toString(),
          max_gpu_type: plan.maxGpuType,
          max_training_time: plan.maxTrainingTime.toString()
        }
      });

      console.log(`✅ Created product: ${product.id}`);

      // Create Stripe price
      const price = await stripe.prices.create({
        product: product.id,
        unit_amount: plan.price,
        currency: 'usd',
        recurring: {
          interval: 'month'
        },
        metadata: {
          plan_name: plan.name
        }
      });

      console.log(`✅ Created price: ${price.id}`);

      // Create or update database record
      await prisma.subscriptionPlan.upsert({
        where: { name: plan.name },
        update: {
          displayName: plan.displayName,
          description: plan.description,
          price: plan.price,
          trainingCredits: plan.trainingCredits,
          maxGpuType: plan.maxGpuType,
          maxTrainingTime: plan.maxTrainingTime,
          features: plan.features,
          stripePriceId: price.id,
          active: true
        },
        create: {
          name: plan.name,
          displayName: plan.displayName,
          description: plan.description,
          price: plan.price,
          currency: 'usd',
          interval: 'month',
          trainingCredits: plan.trainingCredits,
          maxGpuType: plan.maxGpuType,
          maxTrainingTime: plan.maxTrainingTime,
          features: plan.features,
          stripePriceId: price.id,
          active: true
        }
      });

      console.log(`✅ Updated database for ${plan.displayName}`);
    }

    // Create free plan in database only
    await prisma.subscriptionPlan.upsert({
      where: { name: 'free' },
      update: {
        displayName: 'Free Tier',
        description: 'Perfect for getting started with LLM fine-tuning',
        price: 0,
        trainingCredits: 3,
        maxGpuType: 'T4',
        maxTrainingTime: 1800,
        features: [
          '3 training jobs per month',
          'T4 GPU access',
          'Basic models (up to 7B parameters)',
          'Community support',
          'Standard training time (30 min max)',
          'Basic monitoring'
        ],
        stripePriceId: 'free',
        active: true
      },
      create: {
        name: 'free',
        displayName: 'Free Tier',
        description: 'Perfect for getting started with LLM fine-tuning',
        price: 0,
        currency: 'usd',
        interval: 'month',
        trainingCredits: 3,
        maxGpuType: 'T4',
        maxTrainingTime: 1800,
        features: [
          '3 training jobs per month',
          'T4 GPU access',
          'Basic models (up to 7B parameters)',
          'Community support',
          'Standard training time (30 min max)',
          'Basic monitoring'
        ],
        stripePriceId: 'free',
        active: true
      }
    });

    console.log('✅ Created free plan in database');

    console.log('\n🎉 Stripe setup completed successfully!');
    console.log('\nNext steps:');
    console.log('1. Set up webhook endpoint in Stripe dashboard');
    console.log('2. Configure OAuth providers (Google, GitHub)');
    console.log('3. Update STRIPE_PUBLISHABLE_KEY in environment variables');
    console.log('4. Test the payment flow');

  } catch (error) {
    console.error('❌ Error setting up Stripe:', error);
  } finally {
    await prisma.$disconnect();
  }
}

// Run the setup
setupStripeProducts();

