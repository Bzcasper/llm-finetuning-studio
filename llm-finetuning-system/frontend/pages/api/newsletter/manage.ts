import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '../auth/[...nextauth]'
import EmailService from '../../../lib/email'
import { prisma } from '../../../lib/prisma'

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const session = await getServerSession(req, res, authOptions)
  
  if (!session?.user?.email) {
    return res.status(401).json({ message: 'Unauthorized' })
  }

  // Check if user is admin (you can implement your own admin check)
  const user = await prisma.user.findUnique({
    where: { email: session.user.email }
  })

  const isAdmin = user?.email === 'bobby@aitoolpool.com' // Admin check

  if (req.method === 'GET') {
    try {
      // Get newsletter subscribers
      const subscribers = await prisma.user.findMany({
        where: {
          emailSubscribed: true
        },
        select: {
          id: true,
          email: true,
          name: true,
          subscriptionPlan: true,
          createdAt: true
        }
      })

      res.status(200).json({
        subscribers,
        total: subscribers.length
      })
    } catch (error) {
      console.error('Error fetching subscribers:', error)
      res.status(500).json({ message: 'Internal server error' })
    }
  } else if (req.method === 'POST') {
    if (!isAdmin) {
      return res.status(403).json({ message: 'Admin access required' })
    }

    try {
      const { subject, content, targetAudience } = req.body

      if (!subject || !content) {
        return res.status(400).json({ message: 'Subject and content are required' })
      }

      // Get recipients based on target audience
      let whereClause: any = { emailSubscribed: true }

      switch (targetAudience) {
        case 'free':
          whereClause.subscriptionPlan = 'free'
          break
        case 'paid':
          whereClause.subscriptionPlan = { not: 'free' }
          break
        case 'active':
          whereClause.subscriptionStatus = 'ACTIVE'
          break
        case 'all':
        default:
          // No additional filter
          break
      }

      const recipients = await prisma.user.findMany({
        where: whereClause,
        select: {
          email: true,
          name: true
        }
      })

      if (recipients.length === 0) {
        return res.status(400).json({ message: 'No recipients found for the target audience' })
      }

      // Send bulk newsletter
      const emailService = EmailService.getInstance()
      const result = await emailService.sendBulkNewsletter(
        recipients.map(r => ({ email: r.email, name: r.name || 'User' })),
        subject,
        content
      )

      // Log newsletter campaign
      const campaign = await prisma.newsletterCampaign.create({
        data: {
          subject,
          content,
          targetAudience,
          recipientCount: recipients.length,
          sentCount: result.results.filter(r => r.success).length,
          failedCount: result.results.filter(r => !r.success).length,
          status: result.success ? 'sent' : 'failed',
          sentBy: user.id,
        }
      })

      res.status(200).json({
        success: true,
        campaignId: campaign.id,
        recipientCount: recipients.length,
        sentCount: result.results.filter(r => r.success).length,
        failedCount: result.results.filter(r => !r.success).length,
        results: result.results
      })
    } catch (error) {
      console.error('Error sending newsletter:', error)
      res.status(500).json({ message: 'Internal server error' })
    }
  } else if (req.method === 'PUT') {
    // Update subscription preferences
    try {
      const { emailSubscribed } = req.body

      await prisma.user.update({
        where: { email: session.user.email },
        data: { emailSubscribed }
      })

      res.status(200).json({ 
        success: true, 
        message: 'Email preferences updated' 
      })
    } catch (error) {
      console.error('Error updating email preferences:', error)
      res.status(500).json({ message: 'Internal server error' })
    }
  } else {
    res.status(405).json({ message: 'Method not allowed' })
  }
}

