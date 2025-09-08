import { test, expect } from '@playwright/test'

test.describe('Persian Legal AI Dashboard E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should load dashboard with Persian RTL layout', async ({ page }) => {
    // Check RTL direction
    await expect(page.locator('html')).toHaveAttribute('dir', 'rtl')
    
    // Check Persian title
    await expect(page.locator('h1')).toContainText('Persian Legal AI')
    
    // Check navigation menu
    await expect(page.locator('[data-testid="nav-dashboard"]')).toBeVisible()
    await expect(page.locator('[data-testid="nav-team"]')).toBeVisible()
    await expect(page.locator('[data-testid="nav-models"]')).toBeVisible()
  })

  test('should display system metrics correctly', async ({ page }) => {
    // Wait for metrics to load
    await page.waitForSelector('[data-testid="system-metrics"]')
    
    // Check CPU metric
    await expect(page.locator('[data-testid="cpu-metric"]')).toBeVisible()
    
    // Check Memory metric
    await expect(page.locator('[data-testid="memory-metric"]')).toBeVisible()
    
    // Check GPU metric
    await expect(page.locator('[data-testid="gpu-metric"]')).toBeVisible()
  })

  test('should handle navigation between tabs', async ({ page }) => {
    // Click on Team tab
    await page.click('[data-testid="nav-team"]')
    await expect(page.locator('[data-testid="team-content"]')).toBeVisible()
    
    // Click on Models tab
    await page.click('[data-testid="nav-models"]')
    await expect(page.locator('[data-testid="models-content"]')).toBeVisible()
    
    // Click on Analytics tab
    await page.click('[data-testid="nav-analytics"]')
    await expect(page.locator('[data-testid="analytics-content"]')).toBeVisible()
  })

  test('should display training sessions', async ({ page }) => {
    await page.click('[data-testid="nav-models"]')
    
    // Wait for training sessions to load
    await page.waitForSelector('[data-testid="training-sessions"]')
    
    // Check for training session cards
    const sessionCards = page.locator('[data-testid="session-card"]')
    await expect(sessionCards).toHaveCount(1)
    
    // Check session details
    await expect(page.locator('[data-testid="session-status"]')).toContainText('training')
    await expect(page.locator('[data-testid="session-progress"]')).toBeVisible()
  })

  test('should handle real-time updates', async ({ page }) => {
    // Check auto-refresh functionality
    await page.waitForSelector('[data-testid="auto-refresh-toggle"]')
    
    // Toggle auto-refresh
    await page.click('[data-testid="auto-refresh-toggle"]')
    
    // Wait for refresh interval
    await page.waitForTimeout(4000)
    
    // Check that data has been refreshed
    await expect(page.locator('[data-testid="last-updated"]')).toBeVisible()
  })

  test('should be responsive on mobile devices', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    
    // Check sidebar collapse
    await expect(page.locator('[data-testid="sidebar"]')).toHaveClass(/collapsed/)
    
    // Check mobile navigation
    await page.click('[data-testid="mobile-menu-toggle"]')
    await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible()
  })

  test('should handle dark mode toggle', async ({ page }) => {
    // Toggle dark mode
    await page.click('[data-testid="dark-mode-toggle"]')
    
    // Check dark mode classes
    await expect(page.locator('html')).toHaveClass(/dark/)
    
    // Toggle back to light mode
    await page.click('[data-testid="dark-mode-toggle"]')
    await expect(page.locator('html')).not.toHaveClass(/dark/)
  })

  test('should display charts and visualizations', async ({ page }) => {
    await page.click('[data-testid="nav-analytics"]')
    
    // Wait for charts to load
    await page.waitForSelector('[data-testid="performance-chart"]')
    
    // Check chart visibility
    await expect(page.locator('[data-testid="performance-chart"]')).toBeVisible()
    await expect(page.locator('[data-testid="loss-chart"]')).toBeVisible()
    await expect(page.locator('[data-testid="accuracy-chart"]')).toBeVisible()
  })

  test('should handle error states gracefully', async ({ page }) => {
    // Simulate network error
    await page.route('/api/system/health', route => route.abort())
    
    // Reload page
    await page.reload()
    
    // Check error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible()
    
    // Check retry button
    await expect(page.locator('[data-testid="retry-button"]')).toBeVisible()
  })
})