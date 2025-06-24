import { NextApiRequest, NextApiResponse } from 'next'
import Stripe from 'stripe'
import { buffer } from 'micro'
import { prisma } from '../../../lib/prisma'
import EmailService from '../../../lib/email'

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2023-10-16',
})

const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET!

export const config = {
  api: {
    bodyParser: false,
  },
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' })
  }

  const buf = await buffer(req)
  const sig = req.headers['stripe-signature']!

  let event: Stripe.Event

  try {
    event = stripe.webhooks.constructEvent(buf, sig, webhookSecret)
  } catch (err) {
    console.error('Webhook signature verification failed:', err)
    return res.status(400).json({ message: 'Webhook signature verification failed' })
  }

  const emailService = EmailService.getInstance()

  try {
    switch (event.type) {
      case 'checkout.session.completed':
        await handleCheckoutSessionCompleted(event.data.object as Stripe.Checkout.Session, emailService)
        break
      
      case 'customer.subscription.updated':
        await handleSubscriptionUpdated(event.data.object as Stripe.Subscription)
        break
      
      case 'customer.subscription.deleted':
        await handleSubscriptionDeleted(event.data.object as Stripe.Subscription)
        break
      
      case 'invoice.payment_succeeded':
        await handlePaymentSucceeded(event.data.object as Stripe.Invoice, emailService)
        break
      
      case 'invoice.payment_failed':
        await handlePaymentFailed(event.data.object as Stripe.Invoice)
        break
      
      default:
        console.log(`Unhandled event type: ${event.type}`)
    }

    res.status(200).json({ received: true })
  } catch (error) {
    console.error('Error handling webhook:', error)
    res.status(500).json({ message: 'Webhook handler failed' })
  }
}

async function handleCheckoutSessionCompleted(session: Stripe.Checkout.Session, emailService: EmailService) {
  const userId = session.metadata?.userId
  const planId = session.metadata?.planId

  if (!userId || !planId) {
    console.error('Missing metadata in checkout session')
    return
  }

  const plan = await prisma.subscriptionPlan.findUnique({
    where: { id: planId }
  })

  if (!plan) {
    console.error('Plan not found:', planId)
    return
  }

  const user = await prisma.user.findUnique({
    where: { id: userId }
  })

  if (!user) {
    console.error('User not found:', userId)
    return
  }

  // Update user subscription
  await prisma.user.update({
    where: { id: userId },
    data: {
      subscriptionStatus: 'ACTIVE',
      subscriptionPlan: plan.name,
      stripeSubscriptionId: session.subscription as string,
      trainingCredits: {
        increment: plan.trainingCredits
      }
    }
  })

  // Record payment
  await prisma.payment.create({
    data: {
      userId,
      stripePaymentId: session.payment_intent as string,
      amount: session.amount_total || 0,
      currency: session.currency || 'usd',
      status: 'succeeded',
      description: `Subscription to ${plan.displayName}`,
      creditsAdded: plan.trainingCredits,
      subscriptionPlan: plan.name,
    }
  })

  // Send payment confirmation email
  try {
    await emailService.sendPaymentConfirmation(
      user.email,
      user.name || 'User',
      plan.displayName,
      session.amount_total || 0,
      (session.currency || 'usd').toUpperCase()
    )
    console.log(`Payment confirmation email sent to ${user.email}`)
  } catch (error) {
    console.error('Failed to send payment confirmation email:', error)
  }
}

async function handleSubscriptionUpdated(subscription: Stripe.Subscription) {
  const user = await prisma.user.findUnique({
    where: { stripeCustomerId: subscription.customer as string }
  })

  if (!user) {
    console.error('User not found for customer:', subscription.customer)
    return
  }

  const status = subscription.status === 'active' ? 'ACTIVE' : 
                subscription.status === 'past_due' ? 'PAST_DUE' : 
                subscription.status === 'canceled' ? 'CANCELLED' : 'FREE'

  await prisma.user.update({
    where: { id: user.id },
    data: {
      subscriptionStatus: status,
      stripeSubscriptionId: subscription.id,
    }
  })
}

async function handleSubscriptionDeleted(subscription: Stripe.Subscription) {
  const user = await prisma.user.findUnique({
    where: { stripeCustomerId: subscription.customer as string }
  })

  if (!user) {
    console.error('User not found for customer:', subscription.customer)
    return
  }

  await prisma.user.update({
    where: { id: user.id },
    data: {
      subscriptionStatus: 'CANCELLED',
      subscriptionPlan: 'free',
      stripeSubscriptionId: null,
    }
  })
}

async function handlePaymentSucceeded(invoice: Stripe.Invoice, emailService: EmailService) {
  const user = await prisma.user.findUnique({
    where: { stripeCustomerId: invoice.customer as string }
  })

  if (!user) {
    console.error('User not found for customer:', invoice.customer)
    return
  }

  // Add training credits for recurring payments
  const plan = await prisma.subscriptionPlan.findUnique({
    where: { name: user.subscriptionPlan }
  })

  if (plan && invoice.billing_reason === 'subscription_cycle') {
    await prisma.user.update({
      where: { id: user.id },
      data: {
        trainingCredits: {
          increment: plan.trainingCredits
        }
      }
    })

    await prisma.payment.create({
      data: {
        userId: user.id,
        stripePaymentId: invoice.payment_intent as string,
        amount: invoice.amount_paid,
        currency: invoice.currency,
        status: 'succeeded',
        description: `Monthly billing for ${plan.displayName}`,
        creditsAdded: plan.trainingCredits,
        subscriptionPlan: plan.name,
      }
    })

    // Send payment confirmation for recurring payments
    try {
      await emailService.sendPaymentConfirmation(
        user.email,
        user.name || 'User',
        plan.displayName,
        invoice.amount_paid,
        invoice.currency.toUpperCase()
      )
      console.log(`Recurring payment confirmation email sent to ${user.email}`)
    } catch (error) {
      console.error('Failed to send recurring payment confirmation email:', error)
    }
  }
}

async function handlePaymentFailed(invoice: Stripe.Invoice) {
  const user = await prisma.user.findUnique({
    where: { stripeCustomerId: invoice.customer as string }
  })

  if (!user) {
    console.error('User not found for customer:', invoice.customer)
    return
  }

  await prisma.user.update({
    where: { id: user.id },
    data: {
      subscriptionStatus: 'PAST_DUE',
    }
  })
}

