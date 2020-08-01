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
                params={'packageID': 'Specify the ID of the Package'})
@ns_ins_del.route('/insert/<int:packageID>')
class Insert(Resource):
    def post(self, packageID):
        # Inserts a package in the warehouse
        action = warehouse.insertPackage(packageID)
        return {'packageID': packageID, 'insertPos': action}


@ns_ins_del.doc(responses={200: 'Package Successfully Withdrawn'},
                params={'packageID': 'Specify the ID of the Package'})
@ns_ins_del.route('/withdraw/<int:packageID>')
class Withdraw(Resource):
    def post(self, packageID):
        # Withdraws a package in the warehouse
        action = warehouse.insertPackage(packageID)
        return {'packageID': packageID, 'withdrawLos': action}


@ns_ins_del.doc(responses={200: 'Warehouse Successfully Reset'})
@ns_reset.route('/warehouse')
class Reset(Resource):
    def delete(self):
        # Resets the environment of the warehouse
        msg = warehouse.reset()
        return {'status': msg}


if __name__ == '__main__':
    app.run()
