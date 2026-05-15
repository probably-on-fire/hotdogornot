Static HTML for the aired.com root landing page. Tracked in the monorepo so
changes are version-controlled alongside the labeler and training code.
Deployed to `/var/www/aired/html/index.html` on the aired.com server via SFTP.

To deploy from this machine:

```bash
scp aired_site/index.html chris@aired.com:/var/www/aired/html/index.html
```

Or via Python (e.g. from a script without scp in PATH):

```python
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('aired.com', username='chris')          # uses your SSH agent / ~/.ssh/id_*
s = c.open_sftp()
s.put('aired_site/index.html', '/var/www/aired/html/index.html')
s.close(); c.close()
print('deployed')
```
