-- try_reflect (4.0+ per registry): reflect that returns NULL on invocation failure
select try_reflect('java.util.UUID', 'randomUUID') is not null;
select try_reflect('java.util.UUID', 'fromString', 'a5cf6c42-0c85-418f-af6c-3e4e5b1328f2');
select try_reflect('java.lang.Math', 'abs', -1);
select typeof(try_reflect('java.lang.Math', 'abs', -1));
select try_reflect('java.lang.Integer', 'parseInt', 'abc');
select try_reflect('java.lang.Integer', 'parseInt', '12');
select reflect('java.lang.Integer', 'parseInt', 'abc');
select reflect('java.lang.Integer', 'parseInt', '12');
select try_reflect('java.lang.Math', 'floorDiv', 1L, 0L);
select reflect('java.lang.Math', 'floorDiv', 1L, 0L);
select try_reflect('java.lang.Math', 'floorDiv', 7L, 2L);
select try_reflect('java.lang.Math', 'sqrt', 4.0D), try_reflect('java.lang.Math', 'sqrt', -1.0D), try_reflect('java.lang.Math', 'max', 1, 2);
select try_reflect('java.lang.Double', 'toString', double('NaN')), try_reflect('java.lang.Double', 'toString', double('-0.0')), try_reflect('java.lang.Double', 'toString', double('Infinity'));
select try_reflect('java.lang.String', 'valueOf', true), try_reflect('java.lang.String', 'valueOf', cast(1.5 as float));
select try_reflect('java.lang.Long', 'toHexString', 255L);
select try_reflect('java.lang.Math', 'abs', cast(null as int));
select reflect('java.lang.Math', 'abs', cast(null as int));
select try_reflect('java.lang.Integer', 'parseInt', cast(null as string));
select reflect('java.lang.Integer', 'parseInt', cast(null as string));
select try_reflect('java.lang.String', 'valueOf', 1.5BD);
-- unicode / non-utf8 strings
select try_reflect('java.lang.String', 'valueOf', '😀é');
select try_reflect('java.lang.String', 'valueOf', cast(X'ff' as string));
select try_reflect('java.lang.Integer', 'parseInt', ' 12');
-- analysis-time errors (not swallowed by try_)
select try_reflect('java.lang.NoSuchClass', 'x');
select try_reflect('java.lang.Math', 'noSuchMethod', 1);
select try_reflect('java.lang.Math', 'abs', array(1));
select try_reflect('java.lang.Math');
select try_reflect();
select try_reflect(1, 'abs');
select try_reflect('java.lang.Math', 1);
select try_reflect(cast(null as string), 'abs');
select try_reflect('java.lang.Math', cast(null as string));
select try_reflect(concat('java.lang.', 'Math'), 'abs', -1);
create temporary view v as select * from values ('java.lang.Math', -1), ('java.lang.Math', 2) as t(c, x);
select try_reflect(c, 'abs', x) from v;
drop view v;
-- instance method (not static): analysis error, not NULL
select try_reflect('java.lang.String', 'length');
-- table input, empty, conditional, grouping
create temporary view n as select * from values ('12'), ('x'), (null), ('7') as t(s);
select s, try_reflect('java.lang.Integer', 'parseInt', s) from n order by s;
select s, reflect('java.lang.Integer', 'parseInt', s) from n where s is not null and s != 'x' order by s;
select s, case when s = 'x' then -1 else reflect('java.lang.Integer', 'parseInt', s) end from n where s is not null order by s;
select s, if(s = 'x', cast(null as string), reflect('java.lang.Integer', 'parseInt', s)) from n where s is not null order by s;
select coalesce(try_reflect('java.lang.Integer', 'parseInt', s), 'bad') from n order by s;
select try_reflect('java.lang.Integer', 'parseInt', s) from n where s = 'none';
select try_reflect('java.lang.Integer', 'parseInt', s) is null, count(*) from n group by 1 order by 1;
select count(try_reflect('java.lang.Integer', 'parseInt', s)) from n;
drop view n;
-- ANSI off
set spark.sql.ansi.enabled=false;
select try_reflect('java.lang.Integer', 'parseInt', 'abc'), reflect('java.lang.Integer', 'parseInt', '12');
select reflect('java.lang.Integer', 'parseInt', 'abc');
select try_reflect('java.lang.Math', 'abs', 'x');
set spark.sql.ansi.enabled=true;
-- ANSI-on implicit casts of arguments
select try_reflect('java.lang.Math', 'abs', '1');
-- time zone irrelevant; java.lang.System property read
set spark.sql.session.timeZone=America/Los_Angeles;
select try_reflect('java.lang.String', 'valueOf', timestamp'2020-03-08 02:30:00');
select try_reflect('java.lang.System', 'getProperty', 'user.timezone') is not null;
-- function lookup
describe function try_reflect;
describe function extended try_reflect;
