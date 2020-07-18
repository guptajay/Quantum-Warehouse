"""
Implementation of the RESTful API for Quantum Warehouse
"""
from flask import Flask
from flask_restplus import Api, Resource, fields
from deploy import DeployWarehouse

app = Flask(__name__)

api = Api(app, version="1.0",
          title="Quantum Warehouse",
          description="API for Reinforcement Learning architecture to automate long term planning of warehouse inventory for enterprise deployment.",
          doc='/docs')

# Namespaces
ns_ins_del = api.namespace(
    'package', description='Insert and Withdraw Packages')
ns_reset = api.namespace('reset', description='Reset Warehouse Environment')

warehouse = DeployWarehouse()


@ns_ins_del.doc(responses={200: 'Package Successfully Inserted'},
                params={'id': 'Specify the ID of the Package'})
@ns_ins_del.route('/insert/<int:id>')
class Insert(Resource):
    def post(self, id):
        # Inserts a package in the warehouse
        action = warehouse.insertPackage(id)
        return {'packageID': id, 'insertPos': action}


@ns_ins_del.doc(responses={200: 'Package Successfully Withdrawn'},
                params={'id': 'Specify the ID of the Package'})
@ns_ins_del.route('/withdraw/<int:id>')
class Withdraw(Resource):
    def post(self, id):
        # Withdraws a package in the warehouse
        action = warehouse.insertPackage(id)
        return {'packageID': id, 'withdrawLos': action}


@ns_ins_del.doc(responses={200: 'Warehouse Successfully Reset'})
@ns_reset.route('/warehouse')
class Reset(Resource):
    def delete(self):
        # Resets the environment of the warehouse
        msg = warehouse.reset()
        return {'status': msg}


if __name__ == '__main__':
    app.run(debug=True)
