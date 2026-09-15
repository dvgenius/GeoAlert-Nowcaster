const localtunnel = require('localtunnel');
const fs = require('fs');
const path = require('path');

const urlFile = path.join(__dirname, 'tunnel_url.txt');

async function start() {
  try {
    const tunnel = await localtunnel({ port: 5173 });
    const msg = `LOCALTUNNEL ONLINE: ${tunnel.url}\n`;
    fs.writeFileSync(urlFile, tunnel.url, 'utf8');
    console.log('====================================================');
    console.log(msg);
    console.log('====================================================');

    tunnel.on('close', () => {
      console.log('Localtunnel connection closed. Reconnecting in 3 seconds...');
      setTimeout(start, 3000);
    });

    tunnel.on('error', (err) => {
      console.error('Localtunnel error:', err.message);
      tunnel.close();
    });
  } catch (err) {
    console.error('Failed to start localtunnel:', err.message);
    setTimeout(start, 5000);
  }
}

start();
