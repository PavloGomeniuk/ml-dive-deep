const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: '.',
  testMatch: '**/*.spec.js',
  timeout: 30000,
  use: {
    baseURL: process.env.SERVER_URL || 'http://localhost:3000',
    // Use Chromium with fake media streams so we don't need a real microphone.
    // --use-fake-device-for-media-stream: synthetic audio track (sine wave)
    // --use-fake-ui-for-media-stream:     auto-grant mic/camera permissions
    launchOptions: {
      args: [
        '--use-fake-device-for-media-stream',
        '--use-fake-ui-for-media-stream',
      ],
    },
    // Grant microphone and camera to the test origin
    permissions: ['microphone', 'camera'],
  },
});
