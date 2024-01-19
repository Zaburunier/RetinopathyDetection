const { query } = require("express");
const express = require("express");
const app = express();

app.use(express.urlencoded({ extended: false }));
app.use(express.json());
app.set("view engine", "ejs");

const connection = require("./config/db");
app.use(express.static(__dirname + "/public"));
app.use(express.static(__dirname + "/views"));

app.get("/", (req, res) => {
  res.redirect("./index.html");
});

//create operation
app.post("/create", (req, res) => {
  var name = req.body.name;
  var email = req.body.email;
  var phone = req.body.phone;

  connection.query(
    "SELECT COUNT(*) AS CNT FROM ajax_crud_table where email = ? or phone = ?",
    [email, phone],
    (err, count) => {
      if (err) {
        res.send(err);
      } else {
        console.log("count..", count);
        if (count[0].CNT > 0) {
          res.send({ status: 500, message: "Data already present" });
        } else {
          connection.query(
            "INSERT into ajax_crud_table (name,email,phone) values(?,?,?)",
            [name, email, phone],
            function (err, result) {
              if (err) {
                res.send({ status: 500, message: "Something went wrong" });
              } else {
                res.send({ status: 201, message: "Submitted", url: "/read" });
              }
            }
          );
        }
      }
    }
  );
});

//read operation
app.get("/read", (req, res) => {
  connection.query("select * from ajax_crud_table", function (err, rows) {
    if (err) {
      res.send(err);
    } else {
      res.render("read.ejs", { rows });
    }
  });
});

//getting data to the update page
app.get("/update-student", (req, res) => {
  const updateQuery = "select * from ajax_crud_table where id=?";
  connection.query(updateQuery, [req.query.id], (err, eachRow) => {
    if (err) {
      res.send(err);
    } else {
      console.log("update---", eachRow[0]);
      res.render("update.ejs", { data: eachRow[0] });
    }
  });
});

//delete operation
app.delete("/delete_student", (req, res) => {
  var deleted_id = req.body.id;

  var deleteQuery = "delete from ajax_crud_table where id=?";

  connection.query(deleteQuery, deleted_id, (err, rows) => {
    if (err) {
      res.send(err);
    } else {
      connection.query(
        "select count(*) as cnt from ajax_crud_table",
        (err, count) => {
          if (err) {
            res.send(err);
          } else {
            console.log("count", count[0].cnt);
            res.send({ status: 200, message: "Deleted", row: count[0].cnt });
          }
        }
      );
    }
  });
});

//update operation

app.put("/update", (req, res) => {
  const updatedId = req.body.hidden_id;
  const updatedName = req.body.name;
  const updatedEmail = req.body.email;
  const updatedPhone = req.body.phone;

  const updateQuery =
    "update ajax_crud_table set name=?, email=? , phone=? where id=?";

  connection.query(
    updateQuery,
    [updatedName, updatedEmail, updatedPhone, updatedId],
    (err, rows) => {
      if (err) {
        res.send(err);
      } else {
        res.send({ status: 200, url: "/read" });
      }
    }
  );
});

app.listen(process.env.PORT || 3000, (err) => {
  if (err) console.log(err);
  console.log(`connected success on port ${process.env.PORT}`);
});
