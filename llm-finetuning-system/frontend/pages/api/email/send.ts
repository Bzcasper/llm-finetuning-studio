import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '../auth/[...nextauth]'
import EmailService from '../../../lib/email'
import { prisma } from '../../../lib/prisma'

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    
    if (!session?.user?.email) {
      return res.status(401).json({ message: 'Unauthorized' })
    }

    const { type, data } = req.body

    const emailService = EmailService.getInstance()
    let result

    switch (type) {
      case 'welcome':
        result = await emailService.sendWelcomeEmail(
          session.user.email,
          session.user.name || 'User'
        )
        break

      case 'payment_confirmation':
        const { planName, amount, currency } = data
        result = await emailService.sendPaymentConfirmation(
          session.user.email,
          session.user.name || 'User',
          planName,
          amount,
          currency
        )
        break

      case 'training_completed':
        const { modelName, trainingTime, finalLoss } = data
        result = await emailService.sendTrainingCompletedEmail(
          session.user.email,
          session.user.name || 'User',
          modelName,
          trainingTime,
          finalLoss
        )
        break

      case 'newsletter':
        const { subject, content } = data
        result = await emailService.sendNewsletterEmail(
          session.user.email,
          session.user.name || 'User',
          subject,
          content
        )
        break

      default:
        return res.status(400).json({ message: 'Invalid email type' })
    }

    if (result.success) {
      // Log email sent
      await prisma.emailLog.create({
        data: {
          userId: session.user.id,
          type,
          recipient: session.user.email,
          subject: getSubjectForType(type, data),
          status: 'sent',
          resendId: result.id,
        }
      })

      res.status(200).json({ 
        success: true, 
        message: 'Email sent successfully',
        id: result.id 
      })
    } else {
      res.status(500).json({ 
        success: false, 
        message: 'Failed to send email',
        error: result.error 
      })
    }
  } catch (error) {
    console.error('Email API error:', error)
    res.status(500).json({ 
      message: 'Internal server error',
      error: error instanceof Error ? error.message : 'Unknown error'
    })
  }
}

function getSubjectForType(type: string, data: any): string {
  switch (type) {
    case 'welcome':
      return 'Welcome to LLM Fine-Tuning Studio!'
    case 'payment_confirmation':
      return `Payment Confirmation - ${data.planName} Subscription`
    case 'training_completed':
      return `Training Completed - ${data.modelName}`
    case 'newsletter':
      return data.subject
    default:
      return 'LLM Fine-Tuning Studio Notification'
  }
}

