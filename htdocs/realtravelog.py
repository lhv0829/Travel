#!//anaconda3/bin/python3
print("content-type:text/html;")
print()
import cgi
form = cgi.FieldStorage()
print('''<!doctype html>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="author" content="colorlib.com">
    <link href="https://fonts.googleapis.com/css?family=Poppins:400,500,700" rel="stylesheet" />
    <link href="css/main.css" rel="stylesheet" />
  </head>
  <body>
    <div class="s013">
      <form action="process_create.py" method="get">
        <fieldset>
          <legend>Type the Youtube URL</legend>
        </fieldset>
        <div class="inner-form">
          <div class="left">
            <div class="input-wrap first">
              <div class="input-field first">
                <label>URL</label>
                <input type="url" name="link" placeholder="ex: https://www.youtube.com/watch?v=WmP_ncZJCi4" />
              </div>
            </div>
          </div>
          <input class="btn-search" type="submit" value="Search">
        </div>
      </form>
    </div>
  </body>
</html>
''')
